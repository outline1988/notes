*2022.7.6*

### GPIO

#### GPIO的各种模式

##### 输入模式

浮空模式(No pull-up and no pull-down)
	该模式的引脚电平就是真实的外部电压，电平不确定性（单指0或1）。
	[【浮空输入的理解】](https://blog.csdn.net/qq_25814297/article/details/100660837)

上拉输入
下拉输入

[【输入模式的一些特点】](https://blog.csdn.net/changyourmind/article/details/78468883)

HAL库函数
```c
GPIO_PinState HAL_GPIO_ReadPin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);	// 返回枚举的0或1，参数为Port和Pin，适用于输入模式（上拉，下拉，浮空）
// 还有其他的倒是后再查
```

模拟输入：采集的信号直接连接至ADC通道。（和之后的ADC有关，这里跳过）

##### 输出模式

推挽输出：高电平3.3V和低电平0V

开漏输出：低电平必为0V而高电平需要有外部的驱动电路决定。由此开漏输出有以下几个特点：

- 减少MCU内部的驱动
- 通过自己调节外部的上拉电压【来实现不同器件的匹配电平功能。
- 由于需要上拉电阻，故上升沿时会有一定延迟并且功耗增大。

[【开漏输出特点】](https://blog.csdn.net/changyourmind/article/details/78468883)

复用输出模式（开漏和推挽）：与外设有关？不是很懂...
	使用外设来提供驱动。

[【深刻理解GPIO】](https://blog.csdn.net/Seciss/article/details/120595713)

##### GPIO的速度

[【GPIO速度的理解】](https://wenku.baidu.com/view/d1fd206259cfa1c7aa00b52acfc789eb172d9e80.html)



*2022.7.7*

###  例程分析

#### HAL_Init函数

1. 配置预取指缓存Configure Flash Prefetch
   预取指缓存：CPU从Flash中读取指令（Flash时结合了RAM和ROM长处的存储器），FLASH读取指令一次64为，而CPU一次读取32位，故CPU从Flash中读取时需要一个缓冲区，从而提高效率。
2. 其他的暂时不需要了解

#### SystemClock_Config函数

简易时钟树![5F860ADF2C20E9937A76FB02AEBE516B](D:\qq\820936392\FileRecv\MobileFile\5F860ADF2C20E9937A76FB02AEBE516B.png)

stm32通过给每个外设都设置一个时钟使能是为了减小功耗。

#### HAL_Delay

```c
__weak void HAL_Delay(uint32_t Delay)
{
  uint32_t tickstart = HAL_GetTick();
  uint32_t wait = Delay;

  /* Add a freq to guarantee minimum wait */
  if (wait < HAL_MAX_DELAY)
  {
    wait += (uint32_t)(uwTickFreq);
  }

  while((HAL_GetTick() - tickstart) < wait)
  {
  }
}
```

此延迟是通过HAL_GetTick函数获得系统的滴答时间（1ms中断一次）来实现的延迟，并且此函数在Delay的基础上增加1ms来保证最小的延迟时间。

### 外部中断

![image-20220707111412793](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20220707111412793.png)

#### 配置流程（以按键控制LED灯为例）：

- 寻找按键端口，寻找LED灯端口，配置适当的上升下降沿

- 使能NVIC中断通道，不同端口的中断通道不一样。

- 在main.c中重写回调函数
  ```c
  void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
  {
  	if(GPIO_Pin == KEY0_Pin){
  		HAL_GPIO_TogglePin(LED0_GPIO_Port, LED0_Pin);
  	}
  	if(GPIO_Pin == KEY1_Pin){
  		HAL_GPIO_TogglePin(LED1_GPIO_Port, LED1_Pin);
  	}
  }
  // 需要if语句来判断中断的来源是什么端口
  ```

中断流程：中断源 -> 中断源对应函数EXTI3_IRQHandler -> 以此执行HAL_GPIO_EXTI_IRQHandler由此可处理同一中断源的不同端口的打断 -> 清楚中断标志并且进入回调函数（在main.c中定义）

关于中断中使用Delay延时死机的问题，由于Delay采用滴答计时，而系统默认滴答计时中断的优先级为最低，低于外部中断，所以在外部中断中无法使用滴答计时，故而导致死机。
解决方法：

- ~~不使用滴答计时~~；

- 把滴答计时的优先级调高。

- ```c
  for (int i = 0; i < 11550000; ++i) { }	// 精度差不多
  ```



#### NVIC

嵌套向量中断控制器，stm32使用了对应8位寄存器的高4位[7:4]来配置优先级。

优先级分为抢占优先级和子优先级 。

- 抢占优先级：由四位中的高几位进行设置，默认位4位，也就是抢占优先级有0-15共16级，此时子优先级都为0.
- 子优先级：由四位中的后四位表示。
- 作用：当有不同抢占优先级的中断先后发生，且后发生的中断发生时，前一中断还未结束，若后发生的中断抢占优先级较高，则会发生中断嵌套，即暂停前面的中断服务程序，专区后一中断的中断服务程序。子优先级只发生在中断同时发生的情况，若同时发生，则子优先级高的中断先响应。
- 当同一优先级的中断依次出现，后面的中断会等待前面的中断服务程序完成再进入后面的中断。

#### 矩阵键盘

[【矩阵键盘HAL库】](https://blog.csdn.net/weixin_56565733/article/details/122053544)

配置：

- 找GPIO端口，PX0 -> PX7，前四个端口配置为输出，后四个配置为中断，具体配置再思考。（X可代表A->E）
  前四输出低电平（开漏或推挽都可以），后四配置成下降沿触发，默认高电平，下拉电阻。
- 打开中断。
- 添加文件keyboard_exit.h和keyboard_exit.c（目前在dds_wave_gen中的为完整的版本）
- 在回调函数中添加
  ```c
  // extern uint8_t key_times;
  void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
  {
  	KEY_Input_Callback(GPIO_Pin);
  // 	OLED_Show_Int(1, 3, key_times);
  }
  ```

代码详解：

- 中断
  - 后4端口刚开始为下降沿触发，所以中断时可以找到后四中的其中之一。
  - 为了消除按键消抖，检测两次中断的时间间隔，若太短则无视后面的中断。
  - 之后切换GPIO模式，前四位下降沿触发中断，后四位低电平输出，如此可找到前四的端口。
- 发生了中断就用`key_times`变量和`key_values`数组来保存当前的信息，其中`key_times`指向当前元素的下一个元素，`key_values`就是按键信息的数组。
- 可用函数介绍
  - 在主函数中可使用`uint8_t KEY_Input_Char(void)`来获得某一个按键的输入。
  - 在主函数中使用`uint32_t KEY_Input_Num(void)`用来连续输入数字得到整数，需要F键确认。



*2022.7.8*

### 时钟系统

[【RCC时钟详解】](https://blog.csdn.net/as480133937/article/details/98845509)

[【网课：定时器】](https://www.bilibili.com/video/BV1uL4y137Yu)

#### 时钟系统概述

**时钟源**

- HSI：高速内部时钟，集成内部，RC振荡器，精度不高。
- HSE：高速外部时钟，连接至芯片，晶振，或其他时钟源。
- LSI：低速内部时钟，给看门狗用的。
- LSE：低速外部时钟源，必须为32.768kHz的晶振，给RTC提供时钟信号。（RTC实时时钟）

**简易时钟树**![5F860ADF2C20E9937A76FB02AEBE516B](D:\qq\820936392\FileRecv\MobileFile\5F860ADF2C20E9937A76FB02AEBE516B.png)

- 时钟源经过PLL（也可以不经过）成为SYSCLK
- SYSCLK经过AHB（一般为/1）成为HCLK（可由自己设置，由最高限制）。
- HCLK后即可通过各种分频器分给外设时钟信号
  - 内核总线：AHB总线、内核、内存、DMA
  - Tick定时器：经过一定分频（/8或/1）系统定时器时钟
  - I2S总线：不经过分频直接的FCLK
  - APB1和APB2

**APB1和APB2对应外设**

APB2总线：高级定时器timer1, timer8以及通用定时器timer9, timer10, timer11, UTART1, USART6

 APB1总线：通用定时器timer2~timer5，通用定时器timer12~timer14以及基本定时器timer6,timer7  UTART2~UTART5，~~具体不是很懂~~

#### SysTick定时器

**概念：**SysTick也称为滴答定时器，是芯片内部的设备，不属于外设，最终的效果是每隔一定的时间（可自己设置，一般为1ms）产生一个异常，从而切换系统的进程。

**原理：**SysTick定时器首先是一个24位定时器，最大存储数值为2^24。SysTick接收一个时钟信号（一般为系统时钟信号HCLK，分频为1），由初值开始，每一次时钟触发就减一，减到0就产生异常。由于初值可自己设置，故定时器产生异常的间隔也可自己确定，但最大不超过1s，最小不超过1/HCLK。
注：HAL库系统默认的定时器间隔为1ms。

**使用（CubeIDE）：**

- SYS中SysTick打开，作为系统的基准时钟
- NVIC（中断控制台）中设置滴答优先级为0（最高）
- 在stm32f4xx_it.c中的SysTick_Handler函数添加句柄函数HAL_SYSTICK_IRQHandler();
- 最后在main.c重载HAL_SYSTICK_Callback()函数。
  ```c
  /* 滴答定时器中断回调函数 */
  void HAL_SYSTICK_Callback(void) {	// 滴答计时器不需要判断定时器种类
  	static int temp = 0;
  	static int time_value = 0;
  
  	OLED_Show_Int(2, 2, time_value);
  	if (++temp == 1000) {
  		temp = 0;
  		++time_value;
  	}
  }
  ```

  系统的滴答计时器是默认1ms产生一次中断的，所以不需要去关注如何配置各分频系数的问题。

[HAL库中使用滴答计时器](https://blog.csdn.net/qq_16519885/article/details/117756815)

#### HAL_Delay的实现

```c
__weak void HAL_Delay(uint32_t Delay)
{
  uint32_t tickstart = HAL_GetTick();
  uint32_t wait = Delay;

  /* Add a freq to guarantee minimum wait */
  if (wait < HAL_MAX_DELAY)
  {
    wait += (uint32_t)(uwTickFreq);
  }

  while((HAL_GetTick() - tickstart) < wait)
  {
  }
}
```

通过不断获得滴答计时器的数值（单位ms），从而实现ms级别的延时。其本身不为中断，而是对多次中断的计数来实现的。



#### 通用定时器

**定时器中断配置过程**

[【定时器中断1ms一次】](https://blog.csdn.net/PUMBOOOOO/article/details/118219821)

- 设置RCC为HSE外部晶振

- 在系统时钟中查看APB1 Timer clocks的时钟频率

- 定时器TIM3中的源头为内部时钟

- 设置PSC分频系数、上升计数和ARR重装载值（记得减1）
  定时器中断间隔时间溢出时间=重装载值×分频系数/APB1 Timer clocks（这里是84MHz）

- 自动重装载使能

- 打开中断，设置优先级

- main.c中添加回调函数和开始计时函数`HAL_TIM_Base_Start_IT(&htim3)`
  ```c
  /* TIM的中断 */
  void  HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
  	static int temp = 0;
  	static int time_value = 0;
  	if (htim->Instance == TIM3) {	// 注意这里要判断是哪个定时器产生的中断
  
  		OLED_Show_Int(2, 5, time_value);
  //	    HAL_GPIO_TogglePin(LED0_GPIO_Port, LED0_Pin);
  		if (++temp == 1000) {
  			temp = 0;
  			++time_value;
  		}
  	}
  }
  ```

  原理同滴答计数器相同，只不过滴答计数器只能向下计数，而通用计数器可向上、向下和中心计数。


*2022.7.9*

**使用输入捕获来进行低频方波信号的频率检测（小于80kHz）**

[【输入捕获hal库】](https://blog.csdn.net/as480133937/article/details/99407485)

[【有详细代码】](https://blog.csdn.net/azs0504/article/details/119972469)

[【多通道输入捕获】](https://blog.csdn.net/qq_32969455/article/details/107055592)

输入捕获就是对输入信号的上升沿或下降沿或双边沿进行捕获（中断），由此可由计数器记录捕获期间的时间，从而测出方波频率，占空比等参数。

配置（单通道输入捕获）：

- 查看APB1时钟频率（84MHz）
- 选择定时器，开启通道一为输入捕获，打开内部时钟

![在这里插入图片描述](https://img-blog.csdnimg.cn/7ad4b729653f4251893293f6924df7da.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5Y2X5bGx56yR,size_20,color_FFFFFF,t_70,g_se,x_16)

- 中断使能

- 添加代码

  ```c
    /* main.c 中初始化后面加上 */
    HAL_TIM_Base_Start_IT(&htim4);	// 使能中断并启动定时器
  HAL_TIM_IC_Start_IT(&htim4, TIM_CHANNEL_1);
  ```

  ```c
  uint32_t diff = 0;
  uint8_t capture_flag = 0;	// 捕获标志位，0进行第一次，1进行第二次
  uint8_t measure_flag = 0;	// 计算频率标记位，0不计算，1开始计算
  uint32_t cap_val1 = 0;	// 第0位不断记录，第1位暂存第0位的记录
  uint32_t cap_val2 = 0;
  uint16_t overflow_times[2] = {0};
  ```

  ```c
  /* TIM的中断回调 */
  void  HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {	// 用于记录溢出次数
  	if (htim->Instance == TIM4) {
  		++overflow_times[0];
  	}
  }
  ```

  ```c
  /* 输入捕获中断函数 */
  void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim) {
  	if (htim->Instance == TIM4) {
  		if (htim->Channel == HAL_TIM_ACTIVE_CHANNEL_1) {
  			if (capture_flag == 0) {	// 开启第一次捕获
  				cap_val1 = HAL_TIM_ReadCapturedValue(&htim4, TIM_CHANNEL_1);
  				capture_flag = 1;
  				overflow_times[0] = 0;		// 计数值清0
  			}
  			else if (capture_flag == 1) {	// 开启第二次捕获
  				cap_val2 = HAL_TIM_ReadCapturedValue(&htim4, TIM_CHANNEL_1);
  				overflow_times[1] = overflow_times[0];
  				HAL_TIM_IC_Stop_IT(&htim4, TIM_CHANNEL_1);	// 停暂停捕获，防止计算时捕获影响
  				capture_flag = 0;
  				measure_flag = 1;	// 即将开启计算
  
  			}
  			else {
  				Error_Handler();
  			}
  		}
  	}
  }
  ```

  ```c
  /* 计算输入捕获频率，由于相关变量全部为全局变量，故没有参数传参 */
  void cal_capture_freq(void) {
  	if (measure_flag == 1) {
  		diff = overflow_times[1] ? 65536 - cap_val1 + cap_val2 + (overflow_times[1] - 1) * 65536 :
  			                          cap_val2 - cap_val1;
  
  		OLED_Show_Int(70, 5, overflow_times[1]);
  		OLED_Show_Double(2, 5, 84000000 / (double)diff);
  
  		measure_flag = 0;
  		HAL_Delay(1000);	// 延时1s?
  		OLED_Clear();
  
  		HAL_TIM_IC_Start_IT(&htim4, TIM_CHANNEL_1);
  	}
  }
  ```

  上述代码的执行顺序极其关键，一定按照：
  	第一次捕获->溢出值清0->第二次捕获->记录当前溢出值->停止捕获->开启计算，计算完成->开启捕获，进行第一次捕获的流程



补充：带有测量占空比的代码

**配置：在原来的基础上的通道2加上间接输入捕获模式，与直接输入捕获模式的极性相反**

```c
/* 初始化后面加上 */  
HAL_TIM_IC_Start_IT(&htim4, TIM_CHANNEL_2);
```

```c
/* 全局变量 */
uint32_t mid_diff = 0;
uint32_t mid_val = 0;
/* 原来的uint16_t overflow_times[2] = {0};改成下面，即数组多加1位 */
uint16_t overflow_times[3] = {0};	// 0是瞬间值，1是真个周期的溢出值，2是中间的溢出值
```

```c
/* 输入捕获中断函数 */	/* 有一些标志变量需要修改 */
void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim) {
	if (htim->Instance == TIM4) {
		if (htim->Channel == HAL_TIM_ACTIVE_CHANNEL_1) {
			if (capture_flag == 0) {	// 开启第一次捕获
				cap_val1 = HAL_TIM_ReadCapturedValue(&htim4, TIM_CHANNEL_1);
				capture_flag = 1;
				overflow_times[0] = 0;		// 计数值清0
			}
			else if (capture_flag == 2) {	// 开启第三次捕获（发生改变）
				cap_val2 = HAL_TIM_ReadCapturedValue(&htim4, TIM_CHANNEL_1);
				overflow_times[1] = overflow_times[0];
				HAL_TIM_IC_Stop_IT(&htim4, TIM_CHANNEL_1);	// 停暂停捕获，防止计算时捕获影响
				capture_flag = 0;
				measure_flag = 1;	// 即将开启计算

			}
			else {
				Error_Handler();
			}
		}
		else if (htim->Channel == HAL_TIM_ACTIVE_CHANNEL_2) {	// 开启第二次捕获
			if (capture_flag == 1) {
				mid_val = HAL_TIM_ReadCapturedValue(&htim4, TIM_CHANNEL_2);
				overflow_times[2] = overflow_times[0];
				capture_flag = 2;	// 即将开启第三次捕获
			}
		}
	}
}
```

```c
/* 计算输入捕获频率，由于相关变量全部为全局变量，故没有参数传参 */
void cal_capture_freq(void) {
	if (measure_flag == 1) {
		diff = overflow_times[1] ? 65536 - cap_val1 + cap_val2 + (overflow_times[1] - 1) * 65536 :
			                          cap_val2 - cap_val1;
		/* 后面加的 */
		mid_diff = overflow_times[2] ? 65536 - cap_val1 + mid_val + (overflow_times[2] - 1) * 65536 :
									   mid_val - cap_val1;

		OLED_Show_Double(2, 5, 84000000 / (double)diff);
		/* 后面加的 */
		OLED_Show_Double(2, 7, (double)mid_diff / (double)diff);

		measure_flag = 0;
		HAL_Delay(1000);	// 延时1s?
		OLED_Clear();

		HAL_TIM_IC_Start_IT(&htim4, TIM_CHANNEL_1);
	}
}
```

**定时器使用外部时钟源测量频率**

![在这里插入图片描述](https://img-blog.csdnimg.cn/53186f6ecb294124aa95daf657f88f49.png)

只要把时钟源设置为ETR2就可以了，其他和基本定时器配置无异。

思路：使用两个定时器TIM3，使其50ms中断一次，定义TIM2使其的时钟源为待测方波，通过在TIM3这50ms的计数中间，计算计数值CNT的增量，从而计算出待测信号的频率。

**配置流程**

- TIM3作为基本定时器50ms中断一次

  - PSC分频419，ARR为9999（84MHz）
  - 记得打开中断

- TIM2作为外部时钟源（ETR）定时器

  - 只要选择时钟源为ETR就行了
  - PSC为0，ARR最大0xffffffff，保证计数范围够长

- 代码中打开定时器
  ```c
    HAL_TIM_Base_Start_IT(&htim2);
    HAL_TIM_Base_Start_IT(&htim3);
  ```

- 添加其他代码
  ```c
  /* 初始变量的声明 */
  uint8_t interrupt_flag = 0;
  uint8_t measure_flag = 0;
  uint32_t cnt_val1 = 0, cnt_val2 = 0;
  ```

  ```c
  /* 基本定时器50ms中断一次 */
  void  HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
  	if (htim->Instance == TIM3) {
  		/* 内部定时50ms中断一次 */
  		if (interrupt_flag == 0) {
  			cnt_val1 = TIM_ReadBaseValue(&htim2);
  			interrupt_flag = 1;
  		}
  		else if (interrupt_flag == 1) {
  			cnt_val2 = TIM_ReadBaseValue(&htim2);
  			measure_flag = 1;
  		}
  	}
  }
  ```

  ```c
  void cal_interrupt_freq(void) {
  	if (measure_flag == 1) {
  		uint32_t diff = 0;
  		diff = cnt_val1 > cnt_val2 ? 0xffffffff - cnt_val1 + cnt_val2 :
  									 cnt_val2 - cnt_val1;
  		OLED_Show_Double(1, 3, (double)diff * 20);
  		HAL_Delay(1000);
  		OLED_Clear();
  
  		interrupt_flag = 0;
  		measure_flag = 0;
  	}
  }
  ```

  ```c
  /* 在tim.c的文件中 */
  uint32_t TIM_ReadBaseValue(TIM_HandleTypeDef *htim) {
  	return htim->Instance->CNT;
  }
  ```


**两种方式的组合**

思路：

- 由于通过两种方式来分别测量低频段和高频段，所以需要先知道信号属于低频段还是高频段。决定使用两种方式之一来判断频率所属区域。

- 测低频段的频率需要输入捕获和大量的中断，而测高频段只需要读取计数值和基本定时器中断，所以选择高频段的方法。
- 先通过外部触发计数的方法测量频率，若低频则选择输入捕获，高频直接输出。

实际代码参考工程input_capture_final中的cal_freq.c和freq.h

使用说明：

- 首先按照前面两种方法的配置过程。

- 添加中断回调函数
  ```c
  /* htim2外部时钟源ETR，htim3是内部时钟源Base，htim4是输入捕获IC */
  /* TIM3用于中断，TIM2用于计数 */
  void  HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
  	if (htim->Instance == TIM3) {
  		cal_freq_interrupt_Tim_Base_Callback(htim, &htim2);
  	}
  	if (htim->Instance == TIM4) {
  		cal_freq_capture_Tim_Base_Callback();
  	}
  }
  
  /* 输入捕获中断函数 */
  void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim) {
  	cal_freq_interrupt_Tim_IC_Calback(htim);
  }
  ```

- 如此便可直接调用`cal_freq(htim_Base, htim_ETR, htim_IC)`这个函数进行频率计算。（返回double类型的频率）

方波频率测量，完结撒花！！！



### 串口通信

[【重定向串口通信配置】](https://blog.csdn.net/amimax/article/details/124724564)

#### 基础知识

1. 并行通信和串行通信
   - 并行通信占用引脚多，但是速度快。
   - 串行通信占用引脚少，但是速度慢。
2. 串行通信分类
   - 单工：只允许一个方向传输
   - 半双工：允许连个方向传输，但同时只能一个方向
   - 全双工：允许两个方向同时传输

#### CUBEIDE使用串口通讯

**配置流程：**

- usart打开，异步

- 默认波特率

- 一般为8位字长，校验位无，停止位1

- 阻塞模式使用这个函数

  ```c
  HAL_UART_Transmit(&huart1, (uint8_t *)"hello", 5, 0xFFFF);
  HAL_UART_Receive(&huart, (uint8_t *)arr, 5, 0xFFFF);
  // 发送函数没什么可讲的，接收函数被调用时，系统会等待到接收到指定长度的字节时才会退出，会一直等待。
  ```

- 串口调试软件的参数配置和配置的相同

（已经淘汰，实测只能重定向printf）使用重定向：

- 参考lcd_exp工程文件中`Drivers2\Inc\retarget.h`，要使用的时候包含头文件`#include "retarget.h"`即可使用重定向的函数。
- 使能浮点：C/C++ Build -> Settings -> MCU Settings -> 选中两个选项

 新方法（略微复杂）：
[【CubeIDE实现重定向】](https://blog.csdn.net/qq_42212961/article/details/105803129)

- 取消`Core\Src\syscalls.c`的编译
  右键 -> Properties -> C/C++ Build -> Exclude（排除在外，或直接删除）
- 添加文件`retarget.h`和`retarget.c`参考lcd_exp中的Drivers2文件。
- 主函数中`#include "retarget.h"`（此头文件包含了`stdio.h`文件，故主函数中不需要包含）和`Retarget_Init(&huart1);`
- 原理就是重写底层，并把自动配置的底层定义给ban了。。。

**中断模式：**

[【选集：usart中断】](https://www.bilibili.com/video/BV1Ma411v7ZF)

- 选择huart打开，异步，波特率等和阻塞一样

- 打开中断

- 在主函数中使用`HAL_UART_Receive_IT(&huart3, (uint8_t *)&rx, 1);`相当于开始接收中断，当且仅当收到数据时，进行中断。

- 进入中断函数，进行判断即可。
  ```c
  uint8_t rx;
  void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
  	if (huart->Instance == USART3) {
  		if (rx == 0x01) {
  			printf("increase\n");
  		}
  		if (rx == 0x02) {
  			printf("decrease\n");
  		}
  		if (rx == 0x03) {
  			printf("reset\n");
  		}
  		HAL_UART_Receive_IT(huart, (uint8_t *)&rx, 1);	// 开启下次的接收中断
  	}
  }
  ```




### SPI

SPI（Serial Peripheral interface），串行外围设备接口。拥有一主多从、同步通信和高速全双工（几十Mbps）。

#### SPI通讯协议时序

SCK：时钟信号、MISO：主收从发、NSS：片选

SPI通讯协议可通过配置CPOL和CPHA的值来确定：

- CPOL表示默认时钟有效状态，当CPOL为0时，表示静默为时钟低电平，收发数据时为时钟高电平，CPOL为1时则刚好相反。
- CPHA表示触发时许，CPHA为0时，第一个（奇数）跳变沿触发，CPHA为1时，第二个（偶数）跳变沿触发。
- 为了保证在有效跳变沿的时候数据为稳定状态，所以数据改变的有效跳变沿得与之不同。
- 所以一共有两个模式，SCK=0和CPHA=0，上升沿触发收发数据，下降沿改变数据；SCK=1和CPHA=1。

配置流程：

- SPIx选择，IO管脚选择。（2分频，168/2=84MHz）
  - Mode：Full-Duplex Master（一般都是全双工）
  - NSS：Disable
- 选择Motorala方式，8位，MSB高位先行（一般这个）
- CPOL=0，CPHA=0

截止7.21日，oled的spi已改为spi3，详情请看dds_wave_gen2工程。其中OLED_RST为PA12，OLED_DC为PC11，推挽输出，其他默认，然后改以下oled.c文件中的OLED_Write_Byte函数，改为&hspi3，注意SPI3_NSS是随机选择的，所以需要自己手动选择。

### DDS信号发生器（AD9834）

详见dds_wave_gen2工程的`gen_wave.c`和`gen_wave.h`文件。

```c
void gen_wave_SetFreq(uint32_t freq, GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);
void gen_wave_SelectWave(Wave_State Wave, GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);
void gen_wave_GenWave(Wave_State Wave, uint32_t freq, GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);
```

上述函数皆帮我们自动初始化了1kH的正弦波。并且可选择方波正弦波分开。

需要使用信号版继电器和dds的组合。



### DAC

[【hal库DAC配置】](https://blog.csdn.net/as480133937/article/details/102309242)

![DAC配置界面](https://img-blog.csdnimg.cn/20191012100409662.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FzNDgwMTMzOTM3,size_16,color_FFFFFF,t_70)

使用这两个函数

```c
HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, 2048);	// 0-4095	对于0-max
HAL_DAC_Start(&hdac,DAC_CHANNEL_1);	// 开启DAC
```

目前只用stm32输出相对稳定的直流电压，若要波形参数，请使用DDS发生器（GPIO口模拟SPI）。

### DMA

[【一文看懂DMA】](https://blog.csdn.net/as480133937/article/details/104927922)

[【真的手把手教你ADC DMA】](https://blog.csdn.net/tangxianyu/article/details/121149981)

**DMA定义：**在外设和存储器或存储器和存储器之间的高速数据通路，不需要CPU的干预，以减小CPU的资源。

**DMA的传输参数：**

- 数据源地址
- 目标地址
- 数据量
- 进行多少次传输

当剩余数据量为0时传输完毕，否则继续传输。

**DMA传输方式：**

- DMA_Mode_Normal，正常模式：一次DMA传输完，停止DMA，只传输一次。
- DMA_Mode_Circular，循环传输模式：某次传输结束，会自动将传输数据重装，进行下一轮传输，多次传输。
- 仲裁器：确定DMA传输的优先级，会按照外设的优先权进行传输，若优先权相同，则编号小的优先。

**DMA中断：**

传输过半、传输完成和传输错误都可以产生中断。



### ADC

[【ADC框图详解】](https://blog.csdn.net/Firefly_cjd/article/details/108614415)

[【ADC hal库详解 视频】](https://www.bilibili.com/video/BV1mT4y1E7fS)

[【mx配置】](https://blog.csdn.net/zp200003/article/details/121325247)

[【教你各个模式区别 好！】](https://blog.csdn.net/Naisu_kun/article/details/121532288)

**轮询 单通道 单次**

- 选择某个通道
- 扫描模式关闭（这是在多通道的时候使用的）
- 连续模式关闭（因为此时仅仅支持一次开启一次采集）
- 间断模式也要关闭（可能单通道没有影响吧，大概...）

此模式用到

```c
uint32_t adc_value[10] = {0};
for (int i = 0; i < 10; ++i) {
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 50);
    adc_value[i] = HAL_ADC_GetValue(&hadc1);
}
```

每次需要在主程序调用，并且等待采样完毕后才能继续执行，每次采集都要开启ADC一次。

**轮询 单通道 连续**

上述配置中的连续模式打开就行，此模式只要打开ADC，ADC就会一直工作，直到Stop。

```c
uint32_t adc_value[10] = {0};
HAL_ADC_Start(&hadc1);
for (int i = 0; i < 10; ++i) {
    HAL_ADC_PollForConversion(&hadc1, 50);
    adc_value[i] = HAL_ADC_GetValue(&hadc1);
}
HAL_ADC_Stop(&hadc1);
```

其实也可以不使用`HAL_ADC_PollForConversion(&hadc1, 50)`这个函数，但是若不是用，极有可能会发生连续多次读值读到相同的值，原因是采样的时间要大于循环读值的时间间隔。

**轮询 多通道 间断 扫描**

- 扫描模式必定会被打开
- 连续关闭（暂时）
- 间断开启
- 间断分组数1（待查清楚是否为分组）
- 转换次数（Number of Conversion）为打开的通道数目
- Rank1和Rank2用来分别选择打开的通道和采样周期。

```c
for (int i = 0; i < 10; ++i) {
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 50);
    adc_value[i] = HAL_ADC_GetValue(&hadc1);
}
```

每次采样之前也必须先打开ADC和单通道的单词转换很像，但是多通道必须得设置间断模式，否则会出问题。

**轮询 多通道 连续 扫描**

- 连续开启，间断关闭

```c
HAL_ADC_Start(&hadc1);
for (int i = 0; i < 20; ++i) {
    HAL_ADC_PollForConversion(&hadc1, 50);
    adc_value[i] = HAL_ADC_GetValue(&hadc1);
}
HAL_ADC_Stop(&hadc1);
```

**定时器触发的ADC采样（可在一定范围内设置采样频率）：**

- 定时器设置：

  - 首先得在ADC中的触发方式上找到哪个定时器支持Trigger Out event（其他的触发方式待考究）
  - 在符合的定时器设置适合的中断间隔（PSC和ARR）
  - 设置定时器中断
  - 然后先打开中断设置为秒表模式试验一下

- ADC设置：

  - 连续模式不打开
  - 触发方式选择Trigger Out event和Trigger detection on the rising edg。
  - 设置中断

- 程序设置：

  - 打开中断（只有打开了中断才能正常运行）
    ```c
    HAL_TIM_Base_Start_IT(&htim8);
    HAL_ADC_Start_IT(&hadc1);
    ```

  - ```c
    void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)//回调函数
    {
    	static uint8_t i = 0;
    	if (hadc == &hadc1) {
    		adc_value[i] = HAL_ADC_GetValue(hadc);
    		OLED_Show_Int(1, 5, adc_value[i++]);
    	}
    	if (i == 10) { i = 0; }
    }
    ```



#### 定时器触发ADC，DMA采样（1Mhz）

- 定时器设置：

  - TIM8，内部触发
  - PSC = 41，ARR = 1
  - Update Event
  - 打开中断update interrupt

- ADC设置：

  - DMA Continuous Requests开启
  - Tim8 Trigger Out event

- DMA设置：

  - Normal模式
  - 半字节传输
  - DMA默认打开了中断

- 代码：
  ```c
  void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc) {
  	adc_flag = 1;
  	HAL_TIM_Base_Stop(&htim8);
  	HAL_ADC_Stop_DMA(&hadc1);
  }
  ```

  ```c
  HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_value, 100);
  HAL_TIM_Base_Start(&htim8);
  ```

  ```c
  if (adc_flag == 1) {
      adc_flag = 0;
  	/* Start */
      // 在这写你的代码
      /* End */
      adc_value[5];
      HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_value, 100);
      HAL_TIM_Base_Start(&htim8);
  }
  ```

  上述代码的逻辑为：当你要采样时，开启定时器和ADC_DMA，此时的采样依靠定时器触发，此时CPU可以去干其他事情，当数据采集完成，系统会同时中断ADC，此时ADC关闭，数据全部在数组中，需要注意的是，没有完成采集则会导致数组没有采集完全就开始处理而引起错误。

组合：

- 按照上述的方法去配置（定时器触发ADC，DMA采样（1Mhz））

- 添加`col_data.h`和`col_data.c`文件。

- ```c
  void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc) {
  	col_data_ADC_ConvCpltCallback(&hadc1, &htim8);
  }	// 中断回调函数
  ```

- ```c
  get_adc_value(&hadc1, adc_value, 1024, &htim8, 100);	// 在主函数中添加
  // 一定要注意输入的数组的类型是uint16_t
  ```

其他详见adc_exp工程文件。

#### AD7606

基本配置：

- 引脚参考ad7606_exp2工程文件中`Drivers\Inc\AD7606.h`
- 将AD7606.h和AD7606.c驱动文件放入对应文件夹。
- SPI设置：DataSize改为16bit即可，（一定要选择全双工，我也不知道为什么，但是不选就不行，合理怀疑为HAL库的BUG）。

使用：

- 参见ad7606_exp2中的AD7606.h和AD7606.c文件，参考引脚接口。
  主要是需要SPI和PWM，其他GPIO输出即可。（BUSY口开中断）
- 在Core2的文件夹中的col_data_ad7606.c直接使用`get_adc_value_ad7606`，即可。
- 还需要在中断函数中添加：
  ```c
  void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)//PA5 中断
  {
  	col_data_ad7606_GPIO_EXTI_Callback(GPIO_Pin);
  }
  ```
  

PWM设置：

- 主频168Mhz
- 自动装载
- Pulse大概是arr的0.7倍

实测由于接受数据需要23us，所以最高的采样率仅仅能达到40khz，要想提高采样率，就是得考虑spi的dma问题。

### 添加DSP库

[【添加DSP库】](https://blog.csdn.net/mutulu7la/article/details/121056881)

[【浮点数】](https://blog.csdn.net/weixin_43158276/article/details/124268267)

参考dsp_exp工程，将CMSIS_DSP文件夹复制到工程文件中。

- `CMSIS_DSP/Include`添加头文件路径
- `CMSIS_DSP/GCC`添加到库文件路径Library Paths
- Libraries中去头（去掉lib）掐尾（去掉.a）添加库，一般为`arm_cortexM4lf_math`
- Symbols中添加宏定义`ARM_MATH_CM4`
- 设置浮点
- 最后`#include "arm_math.h"`和`#include "arm_const_structs.h"`
- [【DSP库中使用FFT】](https://blog.csdn.net/qq_41529538/article/details/88905039)

**clion中添加dsp库**
[【clion中添加dsp库】](https://www.bilibili.com/read/cv19024271/)

首先打开硬件浮点单元，clion中直接帮我们预写好了`CMakeLists.txt`，可以在`CMakeLists_template.txt`中修改其中的几行
```cmake
#Uncomment for hardware floating point
add_compile_definitions(ARM_MATH_CM4;ARM_MATH_MATRIX_CHECK;ARM_MATH_ROUNDING)
add_compile_options(-mfloat-abi=hard -mfpu=fpv4-sp-d16)
add_link_options(-mfloat-abi=hard -mfpu=fpv4-sp-d16)
```

将接下来的三行取消注释，即可开启FPU。

然后导入DSP库，可以在CubeMX中直接导入，此处神略不写，具体看链接。

再然后，在`CMakeLists.txt`中链接该库，即在`CMakeLists_template.txt`中添加一行代码
```cmake
add_executable($${PROJECT_NAME}.elf $${SOURCES} $${LINKER_SCRIPT})

target_link_libraries(${PROJECT_NAME}.elf ${CMAKE_SOURCE_DIR}/Middlewares/ST/ARM/DSP/Lib/libarm_cortexM4lf_math.a)
```

上述第一行是已经存在的，下面的才是需要添加的。

最后就可以包含`#include "arm_math.h"`从而调用DSP库，注意一定要在`#include "main.h"`之后包含DSP库。

### 串口屏的简单实用

新建工程：

- 选择型号
- 选择方向
- 添加字库，图片

初始化：

- 波特率`bauds=115200`
- `dims=100`
- 在program.s中

背景设置：

- 名字就是page0其实爷就代表着当前页面，在右下角可选择各种选项。

添加图片：

- 在左边栏选择图片（点一下就好，不然会多出很多图片框）
- 可通过右边中的pic选项选择已经加载好的图片。
- 需要注意的是，图片需要提前做好分辨率处理。

添加文本：

- 左边栏选择文本或滚动文本。
- 需要有足够的大小设置才能装下文字。

滑块设置：

- 设置最大最小val。
- 设置初始val。
- 一般在弹起事件中选择滑块达到范围对应的操作。（弹起事件可以选择滑块到达的位置）
- 滑动事件则到达区域直接进行操作。（所以解锁就滑动事件，选择就弹起时间）

按钮设置：

- 谈起事件中写代码即可，一般是跳转某个页面。

串口发送：

`printh 01`即发送十六进制的0x01。

在Cube中的代码：

按照中断的方式配置串口

```c
void use_hmi_Receive_Start(UART_HandleTypeDef *huart);	//使用这个函数开启
```

在中断回调函数中

```c
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
	use_hmi_RxCpltCallback();
}
```

在主函数中
```c
uint8_t use_hmi_check(void)		// 在主函数中使用这个函数阻塞查询
```





测幅频曲线

[【电赛程序】](https://gitee.com/Williamwxh/telegame-library/tree/master/STM32%E7%94%B5%E8%B5%9B%E7%A8%8B%E5%BA%8F)



2022.7电赛最终选择D题，混沌信号发生器，整题没有一点软件部分，上述笔记也没有用。。。

2022.8电赛最终结果为省二等奖，留有遗憾的地方在于运气不佳，测评时没有达到Vpp的百分之八十的指标，不过也挺好的，因为我在划水：）。



### CubeIDE中添加`Core2`和`Drivers2`

首先新建`Core2`和`Drivers2`文件，并在每个文件夹中新建`Inc`和`Src`文件。

添加头文件包含，右键工程文件
![image-20230321203818163](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230321203818163.png)

![image-20230321203850389](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230321203850389.png)

分别选择Includes和Source Location添加头文件`Inc`和源文件`Cores2`和`Drivers2`。
