<div align="center">
    <img src="./doc/icon.png" width="256px"/>
</div>
<div align="center">
    富强、民主、文明、和谐、合规
</div>
<div align="center">
    自由、平等、公正、法治、合法
</div>
<div align="center">
    爱国、敬业、诚信、友善、合理
</div>

# ComfyUI-DeepCache-Fix

## 介绍

ComfyUI加速节点，生成图片时更快，同时保证了加速前后的一致性，适合大批量生图。[Accelerate ComfyUI Nodes for Faster Image Generation, Ensuring Consistency Pre and Post-Acceleration, Ideal for Bulk Image Production.]

## 用法

将该库放在 ComfyUI/custom_nodes/ 下即可。[Simply place the library in the 'ComfyUI/custom_nodes/' directory to get started.]

## 使用说明

### 插件参数

- cache_interval: 缓存间隔, 单位: 步, 默认:3
- cache_depth: 缓存深度, 默认:3
- start_steps: 使用缓存的开始步数, 默认:0
- end_steps: 使用缓存的结束步数, 默认:12
- input_cache: 使用输入层缓存，默认:True 开启
- middle_cahce: 使用中间层缓存，默认:True 开启
- output_cache: 使用输出层缓存，默认:True 开启

### 举例(实践)

#### 模型(蒸馏)

https://www.liblib.art/modelinfo/386109978c19484298d810d6f2830780

#### 生成

在总共16步的执行过程中，我们计划采取分阶段的策略。具体来说，如下

1. 前12步(start_steps=0, end_steps=12)将利用特定的插件来执行，以提高效率和效果。
2. 从第13步开始，我们将切换回原始模型，完成剩下的4步。 (一般预留3步以上，不使用加速)

这样的安排旨在结合两者的优势，确保整个流程的顺利进行。

![img.png](doc%2Fimg.png)

## 参考

原始代码参考: https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86

感谢 laksjdjf 分享的代码。
