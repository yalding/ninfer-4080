#include "ninfer/ops/inkling2_attention.h"
#include "ninfer/ops/inkling2_fused.h"
#include "ninfer/ops/inkling2_moe.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ops/a8_g64_reference.h"
#include "ops/op_tester.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr ReductionCriterion bf16_criterion{1.85e-3,1e-5,3.95e-3};
constexpr ReductionCriterion attn_mma_criterion{6e-3,3e-5,8e-3};
constexpr ReductionCriterion moe_mma_criterion{6e-3,2e-5,1e-2};
constexpr PointwiseCriterion f32_criterion{2e-5,2e-5};
std::vector<float> pattern(std::size_t n,int seed,float scale=1){
    std::vector<float> a(n);fill_uniform(a,seed,-scale,scale);round_to_bf16(a);return a;
}
template<class F>void captured(DeviceContext& ctx,F f){
    cuda_synchronize(); // Order default-stream fixture uploads/guard fills before nonblocking execution.
    f();ctx.synchronize();DecodeGraphDefinition def;DecodeGraphExecutable graph;
    def.capture(ctx.stream,f);graph.instantiate(def);graph.launch(ctx.stream);ctx.synchronize();
}
std::uint16_t half_bits(float x){__half h=__float2half_rn(x);std::uint16_t u;std::memcpy(&u,&h,2);return u;}
double half_value(std::uint16_t u){__half h;std::memcpy(&h,&u,2);return __half2float(h);}
int rel_case(DeviceContext& ctx,int tokens,bool bf){
    auto r=pattern(16*24*tokens,1),bias=pattern(16*1024,2),sink=pattern(24*tokens,3);
    auto dr=to_device(r),db=to_device(bias);DeviceBuffer ds=bf?to_device_bf16(sink):to_device(sink);
    GuardedDeviceBuffer out(std::size_t(1025)*24*tokens*4);
    Tensor rt(dr.p,DType::FP32,{16,24,tokens}),bt(db.p,DType::FP32,{16,1024});
    Tensor st(ds.p,bf?DType::BF16:DType::FP32,{24,tokens}),ot(out.data(),DType::FP32,{1025,24,tokens});
    captured(ctx,[&]{ops::inkling2_rel_bias_prepare(rt,bt,st,ot,ctx.stream);});
    std::vector<double> ref(1025*24*tokens);
    for(int ht=0;ht<24*tokens;++ht)for(int d=0;d<1025;++d){
        double z=-double(sink[ht]);if(d<1024)for(int j=0;j<16;++j)z+=double(r[j+16*ht])*bias[j+16*d];
        ref[d+1025*ht]=z;
    }
    auto got=from_device<float>(out.data(),ref.size());std::vector<double> gd(got.begin(),got.end());
    return verify_pointwise("inkling2 rel table",gd,ref,f32_criterion)+out.verify_guards("rel guards");
}
struct LogicalKV {
    int tokens; std::vector<float> k,v;
    explicit LogicalKV(int count):tokens(count),k(std::size_t(count)*1024),v(k.size()){
        for(int p=0;p<count;++p)for(int h=0;h<8;++h)for(int d=0;d<128;++d){
            auto i=d+128*(h+8*p);
            k[i]=bf16_to_f32(f32_to_bf16(float(((p*7+h*11+d*3)%43)-21)/32));
            v[i]=bf16_to_f32(f32_to_bf16(float(((p*13+h*3+d*5)%47)-23)/32));
        }
    }
};
int attention_oracle(const std::vector<float>& q,const std::vector<float>& gate,
    const std::vector<float>& rel,const LogicalKV& kv,int start,int tokens,bool swa,
    const std::vector<double>& got){
    std::vector<double> ref,actual;
    for(int t=0;t<tokens;++t){
        if(tokens>32 && t!=0 && t!=1 && t!=tokens/2 && t!=tokens-1)continue;
        int pos=start+t,lo=swa?std::max(0,pos-1023):0;
        for(int h=0;h<24;++h){
            if(kv.tokens>32768 && h!=0 && h!=2 && h!=3 && h!=23)continue;
            std::array<double,128> acc{};double maximum=0,den=1;
            for(int p=lo;p<=pos;++p){
                double dot=0;for(int d=0;d<128;++d)dot+=double(q[d+128*(h+24*t)])*kv.k[d+128*(h/3+8*p)];
                double score=dot/std::sqrt(128.0)+rel[std::min(pos-p,1024)+1025*(h+24*t)];
                double next=std::max(maximum,score),old=std::exp(maximum-next),prob=std::exp(score-next);
                den=den*old+prob;
                for(int d=0;d<128;++d)acc[d]=acc[d]*old+prob*kv.v[d+128*(h/3+8*p)];
                maximum=next;
            }
            for(int d=0;d<128;++d){
                ref.push_back(acc[d]/den/(1+std::exp(-double(gate[3072+d+128*h+6528*t]))));
                actual.push_back(got[d+128*(h+24*t)]);
            }
        }
    }
    return verify_reduction(swa?"inkling2 SWA oracle":"inkling2 full oracle",actual,ref,
                            ((!swa&&tokens>=3)||tokens>=16)?attn_mma_criterion:bf16_criterion);
}
void hadamard64_host(double* x){
    for(int step=1;step<64;step<<=1)
        for(int i=0;i<64;i+=2*step)
            for(int j=0;j<step;++j){
                double a=x[i+j],b=x[i+j+step];
                x[i+j]=a+b;x[i+j+step]=a-b;
            }
    for(int i=0;i<64;++i)x[i]*=0.125;
}
int full_case(DeviceContext& ctx,int count,int tokens,KvCacheStorage storage,bool bf16_rel=false){
    int pages=(count+63)/64,start=count-tokens;
    const bool i8=(storage==KvCacheStorage::Int8Group64);
    const bool rk4=(storage==KvCacheStorage::RK4V4E8);
    LogicalKV kv(count);
    auto q=pattern(std::size_t(3072)*tokens,8,0.5f),gate=pattern(std::size_t(6528)*tokens,9,2);
    auto rel=pattern(std::size_t(1025)*24*tokens,10,0.15f);
    std::vector<int> table(pages);for(int p=0;p<pages;++p)table[p]=pages-p-1;
    std::vector<int> positions(tokens);std::iota(positions.begin(),positions.end(),start);
    std::size_t n=std::size_t(pages)*64*1024;
    std::vector<std::uint16_t> kb((!i8&&!rk4)?n:0),vb((!i8&&!rk4)?n:0),ks((i8||rk4)?n/64:0),vs(ks.size());
    std::vector<std::int8_t> kc(i8?n:0),vc(i8?n:0);
    std::vector<std::uint8_t> ku(rk4?n/2:0),vu(rk4?n/2:0);
    for(int p=0;p<count;++p)for(int h=0;h<8;++h)for(int g=0;g<2;++g){
        auto dst=std::size_t(128)*(p%64+64*(h+8*table[p/64]));
        auto si=std::size_t(2)*(p%64+64*(h+8*table[p/64]))+g;
        if(rk4){
            std::array<double,64> rk{},rv{};
            for(int d=0;d<64;++d){auto src=g*64+d+128*(h+8*p);rk[d]=kv.k[src];rv[d]=kv.v[src];}
            hadamard64_host(rk.data());hadamard64_host(rv.data());
            double km=0,vm=0;
            for(int d=0;d<64;++d){km=std::max(km,std::abs(rk[d]));vm=std::max(vm,std::abs(rv[d]));}
            double kscale=half_value(ks[si]=half_bits(float(km/7.0)));
            double vscale=half_value(vs[si]=half_bits(float(vm/7.0)));
            auto udst=std::size_t(64)*(p%64+64*(h+8*table[p/64]))+g*32;
            for(int d=0;d<64;d+=2){
                int c0k=std::clamp(int(std::nearbyint(rk[d]/kscale)),-8,7);
                int c1k=std::clamp(int(std::nearbyint(rk[d+1]/kscale)),-8,7);
                int c0v=std::clamp(int(std::nearbyint(rv[d]/vscale)),-7,7);
                int c1v=std::clamp(int(std::nearbyint(rv[d+1]/vscale)),-7,7);
                ku[udst+d/2]=std::uint8_t((c0k&0xf)|((c1k&0xf)<<4));
                vu[udst+d/2]=std::uint8_t((c0v&0xf)|((c1v&0xf)<<4));
                rk[d]=c0k*kscale;rk[d+1]=c1k*kscale;
                rv[d]=c0v*vscale;rv[d+1]=c1v*vscale;
            }
            hadamard64_host(rk.data());hadamard64_host(rv.data());
            for(int d=0;d<64;++d){auto src=g*64+d+128*(h+8*p);kv.k[src]=float(rk[d]);kv.v[src]=float(rv[d]);}
        }else{
            float km=0,vm=0;
            for(int d=g*64;d<(g+1)*64;++d){auto src=d+128*(h+8*p);km=std::max(km,std::abs(kv.k[src]));vm=std::max(vm,std::abs(kv.v[src]));}
            double kscale=i8?half_value(ks[si]=half_bits(km/127)):0;
            double vscale=i8?half_value(vs[si]=half_bits(vm/127)):0;
            for(int d=g*64;d<(g+1)*64;++d){auto src=d+128*(h+8*p);
                if(i8){
                    kc[dst+d]=std::int8_t(std::clamp(int(std::nearbyint(kv.k[src]/kscale)),-127,127));
                    vc[dst+d]=std::int8_t(std::clamp(int(std::nearbyint(kv.v[src]/vscale)),-127,127));
                    kv.k[src]=float(kc[dst+d]*kscale);kv.v[src]=float(vc[dst+d]*vscale);
                }else{kb[dst+d]=f32_to_bf16(kv.k[src]);vb[dst+d]=half_bits(kv.v[src]);kv.v[src]=float(half_value(vb[dst+d]));}
            }
        }
    }
    DeviceBuffer dk=rk4?to_device(ku):(i8?to_device(kc):to_device(kb));
    DeviceBuffer dv=rk4?to_device(vu):(i8?to_device(vc):to_device(vb));
    DeviceBuffer dks((i8||rk4)?ks.size()*2:0),dvs((i8||rk4)?vs.size()*2:0);
    if(i8||rk4){dks.copy_from_host(ks.data(),ks.size()*2);dvs.copy_from_host(vs.data(),vs.size()*2);}
    auto dt=to_device(table),dp=to_device(positions),dq=to_device_bf16(q),dg=to_device_bf16(gate);
    auto dr=bf16_rel?to_device_bf16(rel):to_device(rel);
    PagedKVLayerView cache;cache.head_dim=128;cache.num_kv_heads=8;
    cache.storage=storage;
    cache.k_pages=Tensor(dk.p,rk4?DType::U8:(i8?DType::I8:DType::BF16),{rk4?64:128,64,8,pages});
    cache.v_pages=Tensor(dv.p,rk4?DType::U8:(i8?DType::I8:DType::FP16),{rk4?64:128,64,8,pages});
    if(i8||rk4){cache.k_scale_pages=Tensor(dks.p,DType::FP16,{2,64,8,pages});cache.v_scale_pages=Tensor(dvs.p,DType::FP16,{2,64,8,pages});}
    cache.block_table=Tensor(dt.p,DType::I32,{pages});
    Tensor qt(dq.p,DType::BF16,{128,24,tokens}),gt(dg.p,DType::BF16,{6528,tokens});
    Tensor rt(dr.p,bf16_rel?DType::BF16:DType::FP32,{1025,24,tokens}),pt(dp.p,DType::I32,{tokens});
    GuardedDeviceBuffer out(std::size_t(tokens)*3072*2);Tensor ot(out.data(),DType::BF16,{128,24,tokens});
    WorkspaceArena workspace(ops::inkling2_full_attention_workspace_capacity_bytes(cache.storage,count,tokens,tokens));
    captured(ctx,[&]{ops::inkling2_full_attention(qt,gt,rt,pt,cache,count,workspace,ot,ctx.stream);});
    int failures=attention_oracle(q,gate,rel,kv,start,tokens,false,from_device_bf16(out.data(),std::size_t(tokens)*3072));
    failures+=out.verify_guards("full guards");failures+=workspace.used()!=0;
    std::cout<<"full keys="<<count<<" T="<<tokens<<" mode="<<(rk4?"rk4":(i8?"i8":"bf16"))<<" failures="<<failures<<'\n';return failures;
}
int swa_case(DeviceContext& ctx,int start,int tokens,bool bf16_rel=false){
    LogicalKV kv(start+tokens);
    auto q=pattern(std::size_t(3072)*tokens,18,0.5f),gate=pattern(std::size_t(6528)*tokens,19,2);
    auto rel=pattern(std::size_t(1025)*24*tokens,20,0.15f);
    std::vector<std::uint16_t> kb(128*1024*8),vb(kb.size());
    for(int p=std::max(0,start-1024);p<start;++p)for(int h=0;h<8;++h)for(int d=0;d<128;++d){
        auto src=d+128*(h+8*p),dst=d+128*((p%1024)+1024*h);
        kb[dst]=f32_to_bf16(kv.k[src]);vb[dst]=half_bits(kv.v[src]);kv.v[src]=float(half_value(vb[dst]));
    }
    std::vector<float> currentk(kv.k.begin()+std::size_t(start)*1024,kv.k.end());
    std::vector<float> currentv(kv.v.begin()+std::size_t(start)*1024,kv.v.end());
    auto dk=to_device(kb),dv=to_device(vb),dq=to_device_bf16(q),dg=to_device_bf16(gate);
    auto dr=bf16_rel?to_device_bf16(rel):to_device(rel);
    auto dck=to_device_bf16(currentk),dcv=to_device_bf16(currentv);
    std::vector<int> positions(tokens);std::iota(positions.begin(),positions.end(),start);auto dp=to_device(positions);
    CyclicKVCacheLayerView cache;cache.capacity=cache.padded_capacity=1024;cache.head_dim=128;cache.num_kv_heads=8;cache.lane_capacity=1;
    cache.k=Tensor(dk.p,DType::BF16,{128,1024,8,1});cache.v=Tensor(dv.p,DType::FP16,{128,1024,8,1});
    Tensor qt(dq.p,DType::BF16,{128,24,tokens}),gt(dg.p,DType::BF16,{6528,tokens});
    Tensor kt(dck.p,DType::BF16,{128,8,tokens}),vt(dcv.p,DType::BF16,{128,8,tokens});
    Tensor rt(dr.p,bf16_rel?DType::BF16:DType::FP32,{1025,24,tokens}),pt(dp.p,DType::I32,{tokens});
    GuardedDeviceBuffer out(std::size_t(tokens)*3072*2);Tensor ot(out.data(),DType::BF16,{128,24,tokens});
    captured(ctx,[&]{ops::inkling2_swa_attention(qt,kt,vt,gt,rt,pt,cache,0,ot,ctx.stream);});
    int failures=attention_oracle(q,gate,rel,kv,start,tokens,true,from_device_bf16(out.data(),std::size_t(tokens)*3072));
    failures+=out.verify_guards("swa guards");
    failures+=verify_exact("swa preserves K ring",from_device<std::uint16_t>(dk,kb.size()),kb);
    failures+=verify_exact("swa preserves V ring",from_device<std::uint16_t>(dv,vb.size()),vb);
    std::cout<<"swa start="<<start<<" T="<<tokens<<" failures="<<failures<<'\n';return failures;
}
int swa_split_case(DeviceContext& ctx,int start,int tokens,bool append){
    LogicalKV kv(start+tokens);
    auto q=pattern(std::size_t(3072)*tokens,18,0.5f),gate=pattern(std::size_t(6528)*tokens,19,2);
    auto rel=pattern(std::size_t(1025)*24*tokens,20,0.15f);
    std::vector<std::uint16_t> kb(128*1024*8),vb(kb.size());
    for(int p=std::max(0,start-1024);p<start;++p)for(int h=0;h<8;++h)for(int d=0;d<128;++d){
        auto src=d+128*(h+8*p),dst=d+128*((p%1024)+1024*h);
        kb[dst]=f32_to_bf16(kv.k[src]);vb[dst]=half_bits(kv.v[src]);kv.v[src]=float(half_value(vb[dst]));
    }
    std::vector<float> currentk(kv.k.begin()+std::size_t(start)*1024,kv.k.end());
    std::vector<float> currentv(kv.v.begin()+std::size_t(start)*1024,kv.v.end());
    auto dk=to_device(kb),dv=to_device(vb),dq=to_device_bf16(q),dg=to_device_bf16(gate),dr=to_device(rel);
    auto dck=to_device_bf16(currentk),dcv=to_device_bf16(currentv);
    std::vector<int> positions(tokens);std::iota(positions.begin(),positions.end(),start);auto dp=to_device(positions);
    CyclicKVCacheLayerView cache;cache.capacity=cache.padded_capacity=1024;cache.head_dim=128;cache.num_kv_heads=8;cache.lane_capacity=1;
    cache.k=Tensor(dk.p,DType::BF16,{128,1024,8,1});cache.v=Tensor(dv.p,DType::FP16,{128,1024,8,1});
    Tensor qt(dq.p,DType::BF16,{128,24,tokens}),gt(dg.p,DType::BF16,{6528,tokens});
    Tensor kt(dck.p,DType::BF16,{128,8,tokens}),vt(dcv.p,DType::BF16,{128,8,tokens});
    Tensor rt(dr.p,DType::FP32,{1025,24,tokens}),pt(dp.p,DType::I32,{tokens});
    GuardedDeviceBuffer out(std::size_t(tokens)*3072*2);Tensor ot(out.data(),DType::BF16,{128,24,tokens});
    WorkspaceArena ws(ops::inkling2_full_attention_workspace_capacity_bytes(KvCacheStorage::RK4V4E8,std::max(1,start+tokens),tokens,tokens));
    captured(ctx,[&]{ops::inkling2_swa_attention(qt,kt,vt,gt,rt,pt,cache,0,static_cast<std::uint32_t>(start+tokens),ws,ot,ctx.stream,append);});
    int failures=attention_oracle(q,gate,rel,kv,start,tokens,true,from_device_bf16(out.data(),std::size_t(tokens)*3072));
    failures+=out.verify_guards("swa split guards");failures+=ws.used()!=0;
    if(append&&tokens==1){
        for(int h=0;h<8;++h)for(int d=0;d<128;++d){
            auto dst=d+128*((start%1024)+1024*h),src=d+128*(h+8*start);
            kb[dst]=f32_to_bf16(kv.k[src]);vb[dst]=half_bits(kv.v[src]);
        }
    }
    failures+=verify_exact("swa split K ring",from_device<std::uint16_t>(dk,kb.size()),kb);
    failures+=verify_exact("swa split V ring",from_device<std::uint16_t>(dv,vb.size()),vb);
    std::cout<<"swa_split start="<<start<<" T="<<tokens<<" append="<<append<<" failures="<<failures<<'\n';return failures;
}

// Sparse test fixtures retain the complete registered expert geometry and decode every
// stored nonzero through an independent byte-stream decoder in the host oracle.
struct SparseQuant {
    bool q3;int rows,k,group;
    std::vector<std::uint8_t> codes;std::vector<std::uint16_t> scales;
    DeviceBuffer dc,ds;Weight view;
    SparseQuant(bool isq3,int r,int width):q3(isq3),rows(r),k(width),group(q3?128:64),
        codes(std::size_t(rows)*k*(q3?3:4)/8),scales(std::size_t(rows)*(k/group),half_bits(0.03125f)) {
        for(int row=0;row<rows;++row)for(int j=0;j<4;++j){
            int col=(row*13+j*197)%k,code=((row+j*3)%7)-3;
            if(q3){auto bit=std::size_t(row)*k*3+col*3;unsigned v=unsigned(code)&7;
                codes[bit/8]|=std::uint8_t(v<<(bit%8));if(bit%8>5)codes[bit/8+1]|=std::uint8_t(v>>(8-bit%8));
            }else{auto i=std::size_t(row)*k+col;codes[i/2]|=std::uint8_t((unsigned(code)&15)<<((i%2)*4));}
        }
        dc=to_device(codes);ds=to_device(scales);
        view.qtype=q3?QType::Q3G128_F16S:QType::Q4G64_F16S;view.qdata=dc.p;view.scales=ds.p;
        view.n=rows;view.k=k;view.group=group;view.group_size=group;view.padded_shape[1]=k;
        view.scale_dtype=DType::FP16;view.layout=QuantLayout::RowSplit;
    }
    double at(int row,int col)const{
        int c;
        if(q3){auto bit=std::size_t(row)*k*3+col*3;unsigned word=codes[bit/8];
            if(bit%8>5)word|=unsigned(codes[bit/8+1])<<8;
            c=(word>>(bit%8))&7;c=c>=4?c-8:c;
        }else{auto i=std::size_t(row)*k+col;c=(codes[i/2]>>((i%2)*4))&15;c=c>=8?c-16:c;}
        return c*half_value(scales[std::size_t(row)*(k/group)+col/group]);
    }
    template<class F>double dot(int row,F value)const{
        double z=0;for(int j=0;j<4;++j){int col=(row*13+j*197)%k;z+=at(row,col)*value(col);}return z;
    }
};
int moe_case(DeviceContext& ctx,bool q3,int tokens){
    SparseQuant gu(q3,128*2304,2304),down(q3,128*2304,1152);
    auto x=pattern(std::size_t(2304)*tokens,30,0.5f);auto dx=to_device_bf16(x);
    auto router=pattern(2304*128,31,0.015625f);auto dr=to_device_bf16(router);
    std::vector<float> bias(128);for(int e=0;e<128;++e)bias[e]=e%13==0?1:0;
    auto db=to_device(bias);
    ops::Inkling2MoeWeights weights;weights.routed_gate_up=gu.view;weights.routed_down=down.view;
    weights.router.qtype=QType::BF16_CTRL;weights.router.n=128;weights.router.k=2304;weights.router.qdata=dr.p;
    weights.exp_probs_bias.qtype=QType::FP32_CTRL;weights.exp_probs_bias.n=128;weights.exp_probs_bias.qdata=db.p;
    Tensor xt(dx.p,DType::BF16,{2304,tokens});GuardedDeviceBuffer out(std::size_t(2304)*tokens*2);
    Tensor ot(out.data(),DType::BF16,{2304,tokens});
    WorkspaceArena workspace(ops::inkling2_moe_workspace_capacity_bytes(gu.view.qtype,tokens,tokens));
    captured(ctx,[&]{ops::inkling2_moe(xt,weights,ot,workspace,ctx.stream);});
    const bool use_a8 = tokens >= 64;
    std::vector<double> x_q;
    if (use_a8) {
        std::vector<std::uint16_t> x_u16(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) x_u16[i] = f32_to_bf16(x[i]);
        x_q = a8_g64_dequantized(x_u16, 2304, tokens);
    }
    std::vector<double> ref(std::size_t(2304)*tokens);
    for(int t=0;t<tokens;++t){
        std::array<double,128> probs{};std::array<int,128> order{};std::iota(order.begin(),order.end(),0);
        for(int e=0;e<128;++e){double dot=0;for(int k=0;k<2304;++k)dot+=double(x[k+2304*t])*router[k+2304*e];probs[e]=1/(1+std::exp(-dot));}
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return probs[a]+bias[a]>probs[b]+bias[b];});
        double norm=1e-20;for(int i=0;i<8;++i)norm+=probs[order[i]];
        for(int i=0;i<8;++i){
            int e=order[i];std::array<double,1152> act{};
            for(int row=0;row<1152;++row){
                double g=gu.dot(e*2304+row,[&](int k){return use_a8?x_q[k+2304*t]:double(x[k+2304*t]);});
                double u=gu.dot(e*2304+row+1152,[&](int k){return use_a8?x_q[k+2304*t]:double(x[k+2304*t]);});
                act[row]=(4*std::tanh(g/4)/(1+std::exp(-g)))*(25*std::tanh(u/25));
            }
            if (use_a8) {
                std::vector<std::uint16_t> act_u16(1152);
                for (int row = 0; row < 1152; ++row) act_u16[row] = f32_to_bf16(static_cast<float>(act[row]));
                auto act_q = a8_g64_dequantized(act_u16, 1152, 1);
                for(int row=0;row<2304;++row)ref[row+2304*t]+=8*probs[e]/norm*down.dot(e*2304+row,[&](int k){return act_q[k];});
            } else {
                for(int row=0;row<2304;++row)ref[row+2304*t]+=8*probs[e]/norm*down.dot(e*2304+row,[&](int k){return act[k];});
            }
        }
    }
    int failures=verify_reduction("inkling2 MoE FP64",from_device_bf16(out.data(),ref.size()),ref,
                                  tokens>=4?moe_mma_criterion:bf16_criterion);
    failures+=out.verify_guards("moe guards");failures+=workspace.used()!=0;
    std::cout<<"moe q3="<<q3<<" T="<<tokens<<" failures="<<failures<<'\n';return failures;
}
int moe_post_sconv_case(DeviceContext& ctx,bool q3,int tokens){
    SparseQuant gu(q3,128*2304,2304),down(q3,128*2304,1152);
    auto x=pattern(std::size_t(2304)*tokens,43,0.4f),router=pattern(128*2304,44,0.015625f);
    std::vector<float> bias(128);for(int e=0;e<128;++e)bias[e]=e%13==0?1:0;
    auto res=pattern(std::size_t(2304)*tokens,46,0.25f);
    auto conv_w=pattern(4*2304,47,0.3f),conv_in=pattern(3*2304,48,0.25f);
    auto dx=to_device_bf16(x),dr=to_device_bf16(router),db=to_device(bias),dres=to_device_bf16(res);
    auto dcw=to_device(conv_w),dcin=to_device_bf16(conv_in);
    ops::Inkling2MoeWeights w;w.routed_gate_up=gu.view;w.routed_down=down.view;
    w.router.qtype=QType::BF16_CTRL;w.router.qdata=dr.p;w.router.n=128;w.router.k=2304;
    w.exp_probs_bias.qtype=QType::FP32_CTRL;w.exp_probs_bias.qdata=db.p;w.exp_probs_bias.n=128;
    Tensor xt(dx.p,DType::BF16,{2304,tokens}),rest(dres.p,DType::BF16,{2304,tokens});
    Tensor cwt(dcw.p,DType::FP32,{4,2304});
    GuardedDeviceBuffer moe_ref_buf(std::size_t(tokens)*2304*2),out_ref_buf(std::size_t(tokens)*2304*2),st_ref_buf(3*2304*2);
    GuardedDeviceBuffer out_fused(std::size_t(tokens)*2304*2),st_fused(3*2304*2),trace_buf(std::size_t(tokens)*2304*2);
    Tensor moe_ref_t(moe_ref_buf.data(),DType::BF16,{2304,tokens}),out_ref_t(out_ref_buf.data(),DType::BF16,{2304,tokens});
    Tensor st_ref_t(st_ref_buf.data(),DType::BF16,{2304,3}),out_fused_t(out_fused.data(),DType::BF16,{2304,tokens});
    Tensor st_fused_t(st_fused.data(),DType::BF16,{2304,3}),trace_t(trace_buf.data(),DType::BF16,{2304,tokens});
    WorkspaceArena ws(ops::inkling2_moe_workspace_capacity_bytes(w.routed_gate_up.qtype,tokens,tokens));
    cuda_synchronize();
    CUDA_CHECK(cudaMemcpyAsync(st_ref_buf.data(),dcin.p,3*2304*2,cudaMemcpyDeviceToDevice,ctx.stream));
    ops::inkling2_moe(xt,w,moe_ref_t,ws,ctx.stream);
    ops::inkling2_fused_sconv(moe_ref_t,cwt,{},st_ref_t,st_ref_t,rest,out_ref_t,ctx.stream);
    ctx.synchronize();
    captured(ctx,[&]{
        CUDA_CHECK(cudaMemcpyAsync(st_fused.data(),dcin.p,3*2304*2,cudaMemcpyDeviceToDevice,ctx.stream));
        ops::inkling2_moe_post_sconv(xt,w,cwt,st_fused_t,st_fused_t,rest,out_fused_t,trace_t,ws,ctx.stream);
    });
    int failures=verify_exact("moe_post_sconv out",from_device<std::uint16_t>(out_fused.data(),std::size_t(tokens)*2304),
                              from_device<std::uint16_t>(out_ref_buf.data(),std::size_t(tokens)*2304));
    failures+=verify_exact("moe_post_sconv state",from_device<std::uint16_t>(st_fused.data(),3*2304),
                           from_device<std::uint16_t>(st_ref_buf.data(),3*2304));
    failures+=verify_exact("moe_post_sconv trace",from_device<std::uint16_t>(trace_buf.data(),std::size_t(tokens)*2304),
                           from_device<std::uint16_t>(moe_ref_buf.data(),std::size_t(tokens)*2304));
    failures+=out_fused.verify_guards("moe_post_sconv guards")+st_fused.verify_guards("moe_post_sconv st guards")+trace_buf.verify_guards("moe_post_sconv trace guards");
    failures+=ws.used()!=0;
    std::cout<<"moe_post_sconv q3="<<q3<<" T="<<tokens<<" failures="<<failures<<'\n';return failures;
}
int dspark_attn_case(DeviceContext& ctx,int context_keys,int tokens,bool swa,KvCacheStorage storage){
    LogicalKV kv(context_keys+tokens);
    auto q=pattern(std::size_t(3072)*tokens,41,0.5f);
    std::vector<float> bk(kv.k.begin()+std::size_t(context_keys)*1024,kv.k.end());
    std::vector<float> bv(kv.v.begin()+std::size_t(context_keys)*1024,kv.v.end());
    auto dq=to_device_bf16(q),dbk=to_device_bf16(bk),dbv=to_device_bf16(bv);
    Tensor qt(dq.p,DType::BF16,{128,24,tokens}),bkt(dbk.p,DType::BF16,{128,8,tokens}),bvt(dbv.p,DType::BF16,{128,8,tokens});
    GuardedDeviceBuffer out(std::size_t(tokens)*3072*2);Tensor ot(out.data(),DType::BF16,{128,24,tokens});
    if(swa){
        std::vector<std::uint16_t> kb(128*1024*8),vb(kb.size());
        for(int p=std::max(0,context_keys-1024);p<context_keys;++p)for(int h=0;h<8;++h)for(int d=0;d<128;++d){
            auto src=d+128*(h+8*p),dst=d+128*((p%1024)+1024*h);
            kb[dst]=f32_to_bf16(kv.k[src]);vb[dst]=half_bits(kv.v[src]);kv.v[src]=float(half_value(vb[dst]));
        }
        auto dk=to_device(kb),dv=to_device(vb);
        CyclicKVCacheLayerView cache;cache.capacity=cache.padded_capacity=1024;cache.head_dim=128;cache.num_kv_heads=8;cache.lane_capacity=1;
        cache.k=Tensor(dk.p,DType::BF16,{128,1024,8,1});cache.v=Tensor(dv.p,DType::FP16,{128,1024,8,1});
        captured(ctx,[&]{ops::inkling2_dspark_swa_attention(qt,bkt,bvt,context_keys,cache,0,ot,ctx.stream);});
    }else{
        int pages=std::max(1,(context_keys+63)/64);
        const bool i8=(storage==KvCacheStorage::Int8Group64),rk4=(storage==KvCacheStorage::RK4V4E8);
        std::vector<int> table(pages);for(int p=0;p<pages;++p)table[p]=pages-p-1;
        std::size_t n=std::size_t(pages)*64*1024;
        std::vector<std::uint16_t> kb((!i8&&!rk4)?n:0),vb((!i8&&!rk4)?n:0),ks((i8||rk4)?n/64:0),vs(ks.size());
        std::vector<std::int8_t> kc(i8?n:0),vc(i8?n:0);
        std::vector<std::uint8_t> ku(rk4?n/2:0),vu(rk4?n/2:0);
        for(int p=0;p<context_keys;++p)for(int h=0;h<8;++h)for(int g=0;g<2;++g){
            auto dst=std::size_t(128)*(p%64+64*(h+8*table[p/64]));
            auto si=std::size_t(2)*(p%64+64*(h+8*table[p/64]))+g;
            if(rk4){
                std::array<double,64> rk{},rv{};
                for(int d=0;d<64;++d){auto src=g*64+d+128*(h+8*p);rk[d]=kv.k[src];rv[d]=kv.v[src];}
                hadamard64_host(rk.data());hadamard64_host(rv.data());
                double km=0,vm=0;for(int d=0;d<64;++d){km=std::max(km,std::abs(rk[d]));vm=std::max(vm,std::abs(rv[d]));}
                double kscale=half_value(ks[si]=half_bits(float(km/7.0))),vscale=half_value(vs[si]=half_bits(float(vm/7.0)));
                auto udst=std::size_t(64)*(p%64+64*(h+8*table[p/64]))+g*32;
                for(int d=0;d<64;d+=2){
                    int c0k=std::clamp(int(std::nearbyint(rk[d]/kscale)),-8,7),c1k=std::clamp(int(std::nearbyint(rk[d+1]/kscale)),-8,7);
                    int c0v=std::clamp(int(std::nearbyint(rv[d]/vscale)),-7,7),c1v=std::clamp(int(std::nearbyint(rv[d+1]/vscale)),-7,7);
                    ku[udst+d/2]=std::uint8_t((c0k&0xf)|((c1k&0xf)<<4));vu[udst+d/2]=std::uint8_t((c0v&0xf)|((c1v&0xf)<<4));
                    rk[d]=c0k*kscale;rk[d+1]=c1k*kscale;rv[d]=c0v*vscale;rv[d+1]=c1v*vscale;
                }
                hadamard64_host(rk.data());hadamard64_host(rv.data());
                for(int d=0;d<64;++d){auto src=g*64+d+128*(h+8*p);kv.k[src]=float(rk[d]);kv.v[src]=float(rv[d]);}
            }else{
                float km=0,vm=0;
                for(int d=g*64;d<(g+1)*64;++d){auto src=d+128*(h+8*p);km=std::max(km,std::abs(kv.k[src]));vm=std::max(vm,std::abs(kv.v[src]));}
                double kscale=i8?half_value(ks[si]=half_bits(km/127)):0,vscale=i8?half_value(vs[si]=half_bits(vm/127)):0;
                for(int d=g*64;d<(g+1)*64;++d){auto src=d+128*(h+8*p);
                    if(i8){kc[dst+d]=std::int8_t(std::clamp(int(std::nearbyint(kv.k[src]/kscale)),-127,127));vc[dst+d]=std::int8_t(std::clamp(int(std::nearbyint(kv.v[src]/vscale)),-127,127));kv.k[src]=float(kc[dst+d]*kscale);kv.v[src]=float(vc[dst+d]*vscale);}
                    else{kb[dst+d]=f32_to_bf16(kv.k[src]);vb[dst+d]=half_bits(kv.v[src]);kv.v[src]=float(half_value(vb[dst+d]));}
                }
            }
        }
        DeviceBuffer dk=rk4?to_device(ku):(i8?to_device(kc):to_device(kb)),dv=rk4?to_device(vu):(i8?to_device(vc):to_device(vb));
        DeviceBuffer dks((i8||rk4)?ks.size()*2:0),dvs((i8||rk4)?vs.size()*2:0);
        if(i8||rk4){dks.copy_from_host(ks.data(),ks.size()*2);dvs.copy_from_host(vs.data(),vs.size()*2);}
        auto dt=to_device(table);
        PagedKVLayerView cache;cache.head_dim=128;cache.num_kv_heads=8;cache.storage=storage;
        cache.k_pages=Tensor(dk.p,rk4?DType::U8:(i8?DType::I8:DType::BF16),{rk4?64:128,64,8,pages});
        cache.v_pages=Tensor(dv.p,rk4?DType::U8:(i8?DType::I8:DType::FP16),{rk4?64:128,64,8,pages});
        if(i8||rk4){cache.k_scale_pages=Tensor(dks.p,DType::FP16,{2,64,8,pages});cache.v_scale_pages=Tensor(dvs.p,DType::FP16,{2,64,8,pages});}
        cache.block_table=Tensor(dt.p,DType::I32,{pages});
        WorkspaceArena ws(ops::inkling2_full_attention_workspace_capacity_bytes(storage,std::max(1,context_keys),tokens,tokens));
        captured(ctx,[&]{ops::inkling2_dspark_full_attention(qt,bkt,bvt,cache,context_keys,ws,ot,ctx.stream);});
    }
    auto got=from_device_bf16(out.data(),std::size_t(tokens)*3072);
    std::vector<double> ref,actual;
    for(int t=0;t<tokens;++t){
        int lo=swa?std::max(0,context_keys+t-1023):0;
        for(int h=0;h<24;++h){
            std::array<double,128> acc{};double maximum=-INFINITY,den=0;
            for(int p=lo;p<context_keys+tokens;++p){
                double dot=0;for(int d=0;d<128;++d)dot+=double(q[d+128*(h+24*t)])*kv.k[d+128*(h/3+8*p)];
                double score=dot/std::sqrt(128.0);
                double next=std::max(maximum,score),old=maximum==-INFINITY?0.0:std::exp(maximum-next),prob=std::exp(score-next);
                den=den*old+prob;
                for(int d=0;d<128;++d)acc[d]=acc[d]*old+prob*kv.v[d+128*(h/3+8*p)];
                maximum=next;
            }
            for(int d=0;d<128;++d){ref.push_back(acc[d]/den);actual.push_back(got[d+128*(h+24*t)]);}
        }
    }
    int failures=verify_reduction("dspark attn",actual,ref,bf16_criterion)+out.verify_guards("dspark guards");
    std::cout<<"dspark swa="<<swa<<" ctx="<<context_keys<<" T="<<tokens<<" failures="<<failures<<'\n';return failures;
}

int rel_bf16_case(DeviceContext& ctx,int t){
    auto r=pattern(16*24*t,701+t,0.25f),b=pattern(16*1024,702,0.25f),s=pattern(24*t,703,0.5f);
    auto dr=to_device(r),db=to_device(b),ds=to_device_bf16(s);
    GuardedDeviceBuffer dt(std::size_t(1025)*24*t*2);
    Tensor tr(dr.p,DType::FP32,{16,24,t}),tb(db.p,DType::FP32,{16,1024}),ts(ds.p,DType::BF16,{24,t}),tt(dt.data(),DType::BF16,{1025,24,t});
    ops::inkling2_rel_bias_prepare(tr,tb,ts,tt,ctx.stream);
    std::vector<double> ref(std::size_t(1025)*24*t);
    for(int tk=0;tk<t;++tk)for(int h=0;h<24;++h){
        const int ht=h+24*tk;const double sk=s[ht];
        for(int d=0;d<1024;++d){double dot=0;for(int i=0;i<16;++i)dot+=double(r[ht*16+i])*b[d*16+i];ref[ht*1025+d]=dot-sk;}
        ref[ht*1025+1024]=-sk;
    }
    auto got_bf=from_device_bf16(dt.data(),ref.size());
    std::vector<double> gd(got_bf.begin(),got_bf.end());
    return verify_pointwise("rel_bf16",gd,ref,PointwiseCriterion{8e-3,2e-2})+dt.verify_guards("rel_bf16 guards");
}

int batched_attn_case(DeviceContext& ctx,KvCacheStorage st,bool swa){
    const int b=4,total_slots=6;
    const std::vector<int> slots_h{2,0,4,1};
    const std::vector<int> pos_h{0,65,1025,2050};
    const int max_pos=*std::max_element(pos_h.begin(),pos_h.end());
    auto q=pattern(128*24*b,801,0.5f),gate=pattern(6528*b,802,0.5f),rel=pattern(1025*24*b,803,0.2f);
    auto dq=to_device_bf16(q),dg=to_device_bf16(gate),dr=to_device(rel),dp=to_device(pos_h),dsl=to_device(slots_h);
    GuardedDeviceBuffer out(128*24*b*2);
    Tensor tq(dq.p,DType::BF16,{128,24,b}),tg(dg.p,DType::BF16,{6528,b}),tr(dr.p,DType::FP32,{1025,24,b}),
        tp(dp.p,DType::I32,{b}),tsl(dsl.p,DType::I32,{b}),tout(out.data(),DType::BF16,{128,24,b});
    WorkspaceArena ws(ops::inkling2_full_attention_workspace_capacity_bytes(st,max_pos+1,b,b));

    std::vector<double> actual,ref;
    int failures = 0;
    if(swa){
        const int hist=1024;
        auto kc=pattern(128*hist*8*total_slots,804,0.5f),vc=pattern(128*hist*8*total_slots,805,0.5f);
        auto ck=pattern(128*8*b,806,0.5f),cv=pattern(128*8*b,807,0.5f);
        std::vector<std::uint16_t> kc_bf(kc.size()),vc_f16(vc.size());
        for(std::size_t i=0;i<kc.size();++i){kc_bf[i]=f32_to_bf16(kc[i]);vc_f16[i]=half_bits(vc[i]);kc[i]=bf16_to_f32(kc_bf[i]);vc[i]=half_value(vc_f16[i]);}
        DeviceBuffer dkc(kc_bf.size()*2),dvc(vc_f16.size()*2);
        dkc.copy_from_host(kc_bf.data(),dkc.bytes);dvc.copy_from_host(vc_f16.data(),dvc.bytes);
        auto dck=to_device_bf16(ck),dcv=to_device_bf16(cv);
        CyclicKVCacheLayerView view{Tensor(dkc.p,DType::BF16,{128,1024,8,total_slots}),Tensor(dvc.p,DType::FP16,{128,1024,8,total_slots}),1024,1024,8,128,total_slots};
        ops::inkling2_swa_attention_batched(tq,Tensor(dck.p,DType::BF16,{128,8,b}),Tensor(dcv.p,DType::BF16,{128,8,b}),tg,tr,tp,tsl,view,max_pos+1,ws,tout,ctx.stream,true);
        ctx.synchronize();
        auto got=from_device_bf16(out.data(),128*24*b);
        for(int i=0;i<b;++i){
            const int sl=slots_h[i],qp=pos_h[i],lo=std::max(0,qp-1023);
            for(int h=0;h<24;++h){
                const int kh=h/3;std::vector<double> acc(128,0.0);double max_s=-1e30,den=0.0;
                const auto ks_fn=[&](int key,int d){return key==qp?double(ck[(i*8+kh)*128+d]):double(kc[((sl*8+kh)*1024+(key&1023))*128+d]);};
                const auto vs_fn=[&](int key,int d){return key==qp?double(cv[(i*8+kh)*128+d]):double(vc[((sl*8+kh)*1024+(key&1023))*128+d]);};
                std::vector<double> sc(qp-lo+1);
                for(int k=lo;k<=qp;++k){double dot=0;for(int d=0;d<128;++d)dot+=double(q[(i*24+h)*128+d])*ks_fn(k,d);sc[k-lo]=dot*0.08838834764831844+rel[(i*24+h)*1025+std::min(qp-k,1024)];max_s=std::max(max_s,sc[k-lo]);}
                const double m_f=std::max(0.0,max_s);den=std::exp(-m_f);
                for(int k=lo;k<=qp;++k){const double p=std::exp(sc[k-lo]-m_f);den+=p;for(int d=0;d<128;++d)acc[d]+=p*vs_fn(k,d);}
                for(int d=0;d<128;++d){const double g=1.0/(1.0+std::exp(-double(gate[i*6528+3072+h*128+d])));ref.push_back(acc[d]/den*g);actual.push_back(got[(i*24+h)*128+d]);}
            }
        }
    }else{
        const int logical_pages=(max_pos+64)/64+1;
        const int phys_pages=logical_pages*total_slots;
        std::vector<int> tables_h(logical_pages*total_slots);
        std::iota(tables_h.begin(),tables_h.end(),0);
        std::reverse(tables_h.begin(),tables_h.end());
        auto k_new=pattern(128*8*b,810,0.5f),v_new=pattern(128*8*b,811,0.5f);
        const bool is_rk4 = st == KvCacheStorage::RK4V4E8 ||
                            st == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64;
        const int is_i8=(st==KvCacheStorage::Int8Group64);
        DeviceBuffer dtbl=to_device(tables_h),dk_new=to_device_bf16(k_new),dv_new=to_device_bf16(v_new);
        DeviceBuffer dk(is_rk4?64*64*8*phys_pages:(is_i8?128*64*8*phys_pages:128*64*8*phys_pages*2));
        DeviceBuffer dv(is_rk4?64*64*8*phys_pages:(is_i8?128*64*8*phys_pages:128*64*8*phys_pages*2));
        DeviceBuffer dks((is_rk4||is_i8)?2*64*8*phys_pages*2:4),dvs((is_rk4||is_i8)?2*64*8*phys_pages*2:4);
        CUDA_CHECK(cudaMemset(dk.p,0,dk.bytes));CUDA_CHECK(cudaMemset(dv.p,0,dv.bytes));
        CUDA_CHECK(cudaMemset(dks.p,0,dks.bytes));CUDA_CHECK(cudaMemset(dvs.p,0,dvs.bytes));
        PagedKVBatchLayerView view{Tensor(dk.p,is_rk4?DType::U8:(is_i8?DType::I8:DType::BF16),{is_rk4?64:128,64,8,phys_pages}),
                                   Tensor(dv.p,is_rk4?DType::U8:(is_i8?DType::I8:DType::FP16),{is_rk4?64:128,64,8,phys_pages}),
                                   (is_rk4||is_i8)?Tensor(dks.p,DType::FP16,{2,64,8,phys_pages}):Tensor{},
                                   (is_rk4||is_i8)?Tensor(dvs.p,DType::FP16,{2,64,8,phys_pages}):Tensor{},
                                   Tensor(dtbl.p,DType::I32,{logical_pages,total_slots}),128,8,st};
        ops::inkling2_kv_append_batched(Tensor(dk_new.p,DType::BF16,{128,8,b}),Tensor(dv_new.p,DType::BF16,{128,8,b}),tp,tsl,view,ctx.stream);
        ops::inkling2_full_attention_batched(tq,tg,tr,tp,tsl,view,max_pos+1,ws,tout,ctx.stream);
        ctx.synchronize();
        auto got=from_device_bf16(out.data(),128*24*b);
        const auto keys = from_device<std::uint8_t>(dk, dk.bytes);
        const auto values = from_device<std::uint8_t>(dv, dv.bytes);
        const auto key_scales = from_device<std::uint16_t>(dks, dks.bytes / 2);
        const auto value_scales = from_device<std::uint16_t>(dvs, dvs.bytes / 2);
        const auto decode = [&](const auto& bytes, const auto& scales, bool key) {
            std::vector<double> decoded(b * 8 * 128);
            for (int i = 0; i < b; ++i) {
                const int pos = pos_h[i];
                const int page = tables_h[slots_h[i] * logical_pages + pos / 64];
                for (int h = 0; h < 8; ++h) {
                    const int row = pos % 64 + 64 * (h + 8 * page);
                    auto* dst = decoded.data() + (i * 8 + h) * 128;
                    for (int d = 0; d < 128; ++d) {
                        if (is_rk4) {
                            const int nibble = (bytes[row * 64 + d / 2] >> ((d & 1) * 4)) & 15;
                            dst[d] = ((nibble ^ 8) - 8) * half_value(scales[2 * row + d / 64]);
                        } else if (is_i8) {
                            const auto code = static_cast<std::int8_t>(bytes[row * 128 + d]);
                            dst[d] = code * half_value(scales[2 * row + d / 64]);
                        } else {
                            std::uint16_t bits;
                            std::memcpy(&bits, &bytes[2 * (row * 128 + d)], sizeof(bits));
                            dst[d] = key ? bf16_to_f32(bits) : half_value(bits);
                        }
                    }
                    if (is_rk4) {
                        hadamard64_host(dst);
                        hadamard64_host(dst + 64);
                    }
                }
            }
            return decoded;
        };
        const auto represented_k = decode(keys, key_scales, true);
        const auto represented_v = decode(values, value_scales, false);
        CUDA_CHECK(cudaMemsetAsync(dk.p, 0, dk.bytes, ctx.stream));
        CUDA_CHECK(cudaMemsetAsync(dv.p, 0, dv.bytes, ctx.stream));
        if (is_rk4 || is_i8) {
            CUDA_CHECK(cudaMemsetAsync(dks.p, 0, dks.bytes, ctx.stream));
            CUDA_CHECK(cudaMemsetAsync(dvs.p, 0, dvs.bytes, ctx.stream));
        }
        for (int i = 0; i < b; ++i) {
            auto* table = static_cast<int*>(dtbl.p) + slots_h[i] * logical_pages;
            const PagedKVLayerView single{view.k_pages, view.v_pages, view.k_scale_pages,
                                          view.v_scale_pages, Tensor(table, DType::I32, {logical_pages}),
                                          128, 8, st};
            const auto row = [i](const DeviceBuffer& buffer) {
                return Tensor(static_cast<std::uint16_t*>(buffer.p) + i * 128 * 8,
                              DType::BF16, {128, 8, 1});
            };
            ops::kv_cache_append(row(dk_new), row(dv_new),
                                 Tensor(static_cast<int*>(dp.p) + i, DType::I32, {1}), single,
                                 ctx.stream);
        }
        ctx.synchronize();
        failures += verify_exact("batched append K bytes", keys, from_device<std::uint8_t>(dk, dk.bytes));
        failures += verify_exact("batched append V bytes", values, from_device<std::uint8_t>(dv, dv.bytes));
        if (is_rk4 || is_i8) {
            failures += verify_exact("batched append K scales", key_scales,
                                     from_device<std::uint16_t>(dks, dks.bytes / 2));
            failures += verify_exact("batched append V scales", value_scales,
                                     from_device<std::uint16_t>(dvs, dvs.bytes / 2));
        }
        for(int i=0;i<b;++i){
            const int qp=pos_h[i];
            for(int h=0;h<24;++h){
                const int kh=h/3;std::vector<double> acc(128,0.0);double max_s=-1e30,den=0.0;
                std::vector<double> sc(qp+1);
                for(int k=0;k<=qp;++k){
                    double dot=0;
                    if(k==qp){for(int d=0;d<128;++d)dot+=double(q[(i*24+h)*128+d])*represented_k[(i*8+kh)*128+d];}
                    sc[k]=dot*0.08838834764831844+rel[(i*24+h)*1025+std::min(qp-k,1024)];max_s=std::max(max_s,sc[k]);
                }
                const double m_f=std::max(0.0,max_s);den=std::exp(-m_f);
                for(int k=0;k<=qp;++k){const double p=std::exp(sc[k]-m_f);den+=p;if(k==qp)for(int d=0;d<128;++d)acc[d]+=p*represented_v[(i*8+kh)*128+d];}
                for(int d=0;d<128;++d){const double g=1.0/(1.0+std::exp(-double(gate[i*6528+3072+h*128+d])));ref.push_back(acc[d]/den*g);actual.push_back(got[(i*24+h)*128+d]);}
            }
        }
    }
    return failures+verify_reduction("batched_attn",actual,ref,attn_mma_criterion)+out.verify_guards("batched_attn guards");
}
}
int main(int argc,char** argv){
    const std::string sel=argc>1?argv[1]:"all";
    try{
        if(cuda_unavailable())return 77;DeviceContext ctx;int failures=0;
        if(sel=="all"||sel=="rel"){
            failures+=rel_case(ctx,1,false);failures+=rel_case(ctx,3,true);failures+=rel_bf16_case(ctx,17);
        }
        if(sel=="all"||sel=="full"){
            for(auto st:{KvCacheStorage::BFloat16,KvCacheStorage::Int8Group64,KvCacheStorage::RK4V4E8})
                for(int count:{64,257,2049})for(int t:{1,8,15,16,17,64,257})if(t<=count)failures+=full_case(ctx,count,t,st);
            for(auto st:{KvCacheStorage::BFloat16,KvCacheStorage::Int8Group64,KvCacheStorage::RK4V4E8})
                failures+=full_case(ctx,257,65,st,true);
            for(auto st:{KvCacheStorage::BFloat16,KvCacheStorage::Int8Group64,KvCacheStorage::RK4V4E8})
                for(int t:{2,3,5,12})failures+=full_case(ctx,1025,t,st);
            for(auto st:{KvCacheStorage::BFloat16,KvCacheStorage::Int8Group64,KvCacheStorage::RK4V4E8})
                failures+=batched_attn_case(ctx,st,false);
            failures+=batched_attn_case(ctx,KvCacheStorage::RotatedInt4KeyInt4ValueGroup64,false);
            failures+=full_case(ctx,131072,1,KvCacheStorage::Int8Group64);
            failures+=full_case(ctx,131072,8,KvCacheStorage::Int8Group64);
            failures+=full_case(ctx,131072,1,KvCacheStorage::RK4V4E8);
            failures+=full_case(ctx,131072,8,KvCacheStorage::RK4V4E8);
        }
        if(sel=="all"||sel=="swa"){
            for(int start:{0,513,1025,2049})for(int t:{1,8,15,16,17,64,129,1024})failures+=swa_case(ctx,start,t);
            failures+=swa_case(ctx,1025,65,true);
            failures+=swa_split_case(ctx,1025,1,true);
            failures+=swa_split_case(ctx,2049,8,false);
            failures+=batched_attn_case(ctx,KvCacheStorage::BFloat16,true);
        }
        if(sel=="all"||sel=="moe"){
            for(bool q3:{false,true})for(int t:{1,3,4,8,15,16,17,63,64,65,128,256,1024})failures+=moe_case(ctx,q3,t);
            for(bool q3:{false,true})for(int t:{1,4,8,16})failures+=moe_post_sconv_case(ctx,q3,t);
        }
        if(sel=="all"||sel=="dspark"){
            for(int ctx_k:{0,65,1025})failures+=dspark_attn_case(ctx,ctx_k,8,true,KvCacheStorage::BFloat16);
            for(auto st:{KvCacheStorage::BFloat16,KvCacheStorage::Int8Group64,KvCacheStorage::RK4V4E8})
                for(int ctx_k:{0,257})failures+=dspark_attn_case(ctx,ctx_k,8,false,st);
        }
        std::cout<<"inkling2 attention/MoE failures="<<failures<<'\n';return failures?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
