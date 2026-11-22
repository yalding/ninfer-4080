add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)
if(WIN32)
  set_target_properties(ninfer PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)
if(WIN32)
  set_target_properties(ninfer-serve PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
target_include_directories(ninfer-perplexity PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/perplexity)
target_link_libraries(ninfer-perplexity PRIVATE ninfer_engine ninfer_product_logging)
if(WIN32)
  set_target_properties(ninfer-perplexity PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()
