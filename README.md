# vit.cpp
一个用于学习的C++实现的基于ggml的vit模型推理实现，主要由于原本的项目无法编译，并且没有进行更新，导致无法使用
所以基于原始的版本进行了一个新的实现并简化，使得代码更易读

# build
```
cmake -B build
cmake --build build -j --config Release
```

# run inference
```
vit-ctx.exe -m ./ggml-model-f16.gguf -i ./assets/apple.jpg
```

# dependency
* [vit.cpp] (https://github.com/staghado/vit.cpp)
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
