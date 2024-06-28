# vit.cpp
一个用于学习的C++实现的基于ggml的vit模型推理实现，主要由于原本的项目无法编译，并且没有进行更新，导致无法使用
所以基于原始的版本进行了一个新的实现并简化，使得代码更易读

# convert model
clone本项目
```
git clone --recurse-submodules https://github.com/staghado/vit.cpp.git
```

convert model
````
python convert-pth-to-ggml.py --model_name xxxxx
```

# build
```
cmake -B build
cmake --build build -j --config Release
```

# run inference
```
vit-ctx.exe -m ./ggml-model-f16.gguf -i ./assets/apple.jpg
```

# result
```
main: seed = -1
vit_model_load: loading model from './ggml-model-f16.gguf' - please wait
vit_model_load: hidden_size            = 768
vit_model_load: num_hidden_layers      = 12
vit_model_load: num_attention_heads    = 12
vit_model_load: patch_size             = 8
vit_model_load: img_size               = 224
vit_model_load: num_classes            = 1000
vit_model_load: ftype                  = 1
vit_model_load: qntvr                  = 0
operator (): ggml ctx size = 166.97 MB
vit_model_load: ................... done
vit_model_load: model size =   166.52 MB / num tensors = 152
main: load image from: ./polars.jpeg, image size: [286 x 176]
newWidth, newHeight = 224, 224
tx, ty = 1.276786, 0.785714
nx, ny = 286, 176
vit_image_preprocess_bicubic: scale = 1.276786
main: processed image: ./polars.jpeg, image size: [224 x 224]
predict label: giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca, prob: 0.876688

main:    model load time =   131.43 ms
main:    processing time =  1974.41 ms
main:    total time      =  2105.84 ms
```

# dependency
* [vit.cpp](https://github.com/staghado/vit.cpp)
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
