# 太极图形课S1-fem弹性物体仿真加渲染

## 渲染效果 
[渲染视频](https://www.bilibili.com/video/BV1zh411x7za/)
## 依赖模块
taichi、numpy、pillow、meshio、opencv-python ；另外为了仿真，制作四面体剖分网格的模型，使用了TetGen工具；具体是将原始的obj文件转化为stl格式，再用TetGen工具将stl转化为mesh格式的。
 
## 执行步骤 
1、模拟仿真生成一系列obj文件  
   参考命令： python fem_main.py  
2、将上一步生成的obj在cornellbox模型场景中渲染生成png图片  
   参考命令：python render_main.py  
3、合成mp4格式的视频  
   参考命令:python make_video.py  
   
## 说明 
1、由于上面第二步执行比较慢（我的电脑平均一张图片要渲染30s）并且每渲染7张图片我电脑就会报错（后续研究下看能否解决），执行终止；所以如果想渲染几百张图片要慎重考虑，  
可以降低一下最终合成视频的帧率，比如24，这样最终可以少渲染一些。  
2、上面的渲染视频看起来好像是慢动作播放，是不是我合成视频的帧率故意设少了？其实不是，我是根据仿真的time step计算出的每秒将近60的帧率，看起来效果像慢动作的原因是空间距离的单位，  
仿真中模型运动的单位是“米”,而要和康纳尔盒子的空间单位一致的话，盒子天花板到地板的距离就是500多米了，所以合成的视频看起来像慢动作，不够物理真实了。  
3、可以调整下杨氏模量等相关系数以及不同的四面体网格细分程度来达到不同的落地碰撞效果，也可以在初始状态给模型所有点、或某些点初速度来得到不同的运动效果；不过四周墙壁的碰撞边界条件要自己实现了；
