#include "ninfer/ops/inkling2_attention.h"
#include "ninfer/ops/inkling2_moe.h"
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
                            tokens>=16?attn_mma_criterion:bf16_criterion);
}
int full_case(DeviceContext& ctx,int count,int tokens,bool i8){
    int pages=(count+63)/64,start=count-tokens;
    LogicalKV kv(count);
    auto q=pattern(std::size_t(3072)*tokens,8,0.5f),gate=pattern(std::size_t(6528)*tokens,9,2);
    auto rel=pattern(std::size_t(1025)*24*tokens,10,0.15f);
    std::vector<int> table(pages);for(int p=0;p<pages;++p)table[p]=pages-p-1;
    std::vector<int> positions(tokens);std::iota(positions.begin(),positions.end(),start);
    std::size_t n=std::size_t(pages)*64*1024;
    std::vector<std::uint16_t> kb(i8?0:n),vb(i8?0:n),ks(i8?n/64:0),vs(ks.size());
    std::vector<std::int8_t> kc(i8?n:0),vc(kc.size());
    for(int p=0;p<count;++p)for(int h=0;h<8;++h)for(int g=0;g<2;++g){
        auto dst=std::size_t(128)*(p%64+64*(h+8*table[p/64]));
        auto si=std::size_t(2)*(p%64+64*(h+8*table[p/64]))+g;
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
    DeviceBuffer dk=i8?to_device(kc):to_device(kb),dv=i8?to_device(vc):to_device(vb);
    DeviceBuffer dks(i8?ks.size()*2:0),dvs(i8?vs.size()*2:0);
    if(i8){dks.copy_from_host(ks.data(),ks.size()*2);dvs.copy_from_host(vs.data(),vs.size()*2);}
    auto dt=to_device(table),dp=to_device(positions),dq=to_device_bf16(q),dg=to_device_bf16(gate),dr=to_device(rel);
    PagedKVLayerView cache;cache.head_dim=128;cache.num_kv_heads=8;
    cache.storage=i8?KvCacheStorage::Int8Group64:KvCacheStorage::BFloat16;
    cache.k_pages=Tensor(dk.p,i8?DType::I8:DType::BF16,{128,64,8,pages});
    cache.v_pages=Tensor(dv.p,i8?DType::I8:DType::FP16,{128,64,8,pages});
    if(i8){cache.k_scale_pages=Tensor(dks.p,DType::FP16,{2,64,8,pages});cache.v_scale_pages=Tensor(dvs.p,DType::FP16,{2,64,8,pages});}
    cache.block_table=Tensor(dt.p,DType::I32,{pages});
    Tensor qt(dq.p,DType::BF16,{128,24,tokens}),gt(dg.p,DType::BF16,{6528,tokens});
    Tensor rt(dr.p,DType::FP32,{1025,24,tokens}),pt(dp.p,DType::I32,{tokens});
    GuardedDeviceBuffer out(std::size_t(tokens)*3072*2);Tensor ot(out.data(),DType::BF16,{128,24,tokens});
    WorkspaceArena workspace(ops::inkling2_full_attention_workspace_capacity_bytes(cache.storage,count,tokens,tokens));
    captured(ctx,[&]{ops::inkling2_full_attention(qt,gt,rt,pt,cache,count,workspace,ot,ctx.stream);});
    int failures=attention_oracle(q,gate,rel,kv,start,tokens,false,from_device_bf16(out.data(),std::size_t(tokens)*3072));
    failures+=out.verify_guards("full guards");failures+=workspace.used()!=0;
    std::cout<<"full keys="<<count<<" T="<<tokens<<" i8="<<i8<<" failures="<<failures<<'\n';return failures;
}
int swa_case(DeviceContext& ctx,int start,int tokens){
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
    captured(ctx,[&]{ops::inkling2_swa_attention(qt,kt,vt,gt,rt,pt,cache,0,ot,ctx.stream);});
    int failures=attention_oracle(q,gate,rel,kv,start,tokens,true,from_device_bf16(out.data(),std::size_t(tokens)*3072));
    failures+=out.verify_guards("swa guards");
    failures+=verify_exact("swa preserves K ring",from_device<std::uint16_t>(dk,kb.size()),kb);
    failures+=verify_exact("swa preserves V ring",from_device<std::uint16_t>(dv,vb.size()),vb);
    std::cout<<"swa start="<<start<<" T="<<tokens<<" failures="<<failures<<'\n';return failures;
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
    std::vector<double> ref(std::size_t(2304)*tokens);
    for(int t=0;t<tokens;++t){
        std::array<double,128> probs{};std::array<int,128> order{};std::iota(order.begin(),order.end(),0);
        for(int e=0;e<128;++e){double dot=0;for(int k=0;k<2304;++k)dot+=double(x[k+2304*t])*router[k+2304*e];probs[e]=1/(1+std::exp(-dot));}
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return probs[a]+bias[a]>probs[b]+bias[b];});
        double norm=1e-20;for(int i=0;i<8;++i)norm+=probs[order[i]];
        for(int i=0;i<8;++i){
            int e=order[i];std::array<double,1152> act{};
            for(int row=0;row<1152;++row){
                double g=gu.dot(e*2304+row,[&](int k){return x[k+2304*t];});
                double u=gu.dot(e*2304+row+1152,[&](int k){return x[k+2304*t];});
                act[row]=(4*std::tanh(g/4)/(1+std::exp(-g)))*(25*std::tanh(u/25));
            }
            for(int row=0;row<2304;++row)ref[row+2304*t]+=8*probs[e]/norm*down.dot(e*2304+row,[&](int k){return act[k];});
        }
    }
    int failures=verify_reduction("inkling2 MoE FP64",from_device_bf16(out.data(),ref.size()),ref,
                                  tokens>=16?moe_mma_criterion:bf16_criterion);
    failures+=out.verify_guards("moe guards");failures+=workspace.used()!=0;
    std::cout<<"moe q3="<<q3<<" T="<<tokens<<" failures="<<failures<<'\n';return failures;
}
}
int main(int argc,char** argv){
    const std::string sel=argc>1?argv[1]:"all";
    try{
        if(cuda_unavailable())return 77;DeviceContext ctx;int failures=0;
        if(sel=="all"||sel=="rel"){
            failures+=rel_case(ctx,1,false);failures+=rel_case(ctx,3,true);
        }
        if(sel=="all"||sel=="full"){
            for(bool i8:{false,true})for(int count:{64,257,2049})for(int t:{1,15,16,17,64,257})if(t<=count)failures+=full_case(ctx,count,t,i8);
            failures+=full_case(ctx,131072,1,true);
        }
        if(sel=="all"||sel=="swa"){
            for(int start:{0,513,1025,2049})for(int t:{1,15,16,17,64,129,1024})failures+=swa_case(ctx,start,t);
        }
        if(sel=="all"||sel=="moe"){
            for(bool q3:{false,true})for(int t:{1,3,15,16,17,64,65,256,1024})failures+=moe_case(ctx,q3,t);
        }
        std::cout<<"inkling2 attention/MoE failures="<<failures<<'\n';return failures?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
