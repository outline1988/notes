### 11. Timers

#### 11.1 Introduce to Timers

先看完后面的再来总结这里

#### 11.2 Basic Timers

基本定时器所有其他定时器的基础，所有的定时器都拥有基本定时器的功能，例如，HAL库中定时器的函数`HAL_TIM_Base_XXX`虽然以基本定时器命名，但是适用于所有其他定时器。

HAL库中用一个`TIM_HandleTypeDef`类型的结构体来表示一个定时器，其简化定义如下

```c
typedef struct
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
{
  TIM_TypeDef                        *Instance;         /*!< Register base address */
  TIM_Base_InitTypeDef               Init;              /*!< TIM Time Base required parameters */
  HAL_TIM_ActiveChannel              Channel;           /*!< Active channel */
  DMA_HandleTypeDef                  *hdma[7];          /*!< DMA Handlers array                                                             					This array is accessed by a @ref DMA_Handle_index */
  HAL_LockTypeDef                    Lock;              /*!< Locking object  */
  __IO HAL_TIM_StateTypeDef          State;             /*!< TIM operation state */                             
} TIM_HandleTypeDef;
```

- Instance：与其他外设的句柄一样，是一个指向该外设寄存器的指针；

- Init：同样与其他外设一样，其为定时器的初始化参数；

- Channel：是一个枚举变量类型，用于指示该定时器的哪些通道是活跃状态。

  ```c
  typedef enum {
    HAL_TIM_ACTIVE_CHANNEL_1        = 0x01U,    /*!< The active channel is 1     */
    HAL_TIM_ACTIVE_CHANNEL_2        = 0x02U,    /*!< The active channel is 2     */
    HAL_TIM_ACTIVE_CHANNEL_3        = 0x04U,    /*!< The active channel is 3     */
    HAL_TIM_ACTIVE_CHANNEL_4        = 0x08U,    /*!< The active channel is 4     */
    HAL_TIM_ACTIVE_CHANNEL_CLEARED  = 0x00U     /*!< All active channels cleared */
  } HAL_TIM_ActiveChannel;
  ```

- *hdma[7]：这是一个七个元素的数组，其中每个元素都是指向`DMA_HandleTypeDef`类型的指针（复习C和指针，下标的优先级要比接引用高，所以先对其下标，在对其解引用）；

- State：用于追踪当前定时器的状态。

所有的定时器（包括除基本定时器以外的定时器）初始化配置都是通过配置类型`TIM_Base_InitTypeDef`来进行的，其定义如下

```c
typedef struct {
  uint32_t Prescaler;         /*!< Specifies the prescaler value used to divide the TIM clock.
                                   This parameter can be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF */

  uint32_t CounterMode;       /*!< Specifies the counter mode.
                                   This parameter can be a value of @ref TIM_Counter_Mode */

  uint32_t Period;            /*!< Specifies the period value to be loaded into the active
                                   Auto-Reload Register at the next update event.
                                   This parameter can be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF.  */

  uint32_t ClockDivision;     /*!< Specifies the clock division.
                                   This parameter can be a value of @ref TIM_ClockDivision */

  uint32_t RepetitionCounter;  /*!< Specifies the repetition counter value. Each time the RCR downcounter
                                    reaches zero, an update event is generated and counting restarts
                                    from the RCR value (N).
                                    This means in PWM mode that (N+1) corresponds to:
                                        - the number of PWM periods in edge-aligned mode
                                        - the number of half PWM period in center-aligned mode
                                     GP timers: this parameter must be a number between Min_Data = 0x00 and
                                     Max_Data = 0xFF.
                                     Advanced timers: this parameter must be a number between Min_Data = 0x0000 and
                                     Max_Data = 0xFFFF. */

  uint32_t AutoReloadPreload;  /*!< Specifies the auto-reload preload.
                                   This parameter can be a value of @ref TIM_AutoReloadPreload */
} TIM_Base_InitTypeDef;
```

- Prescaler：该参数将定时器的时钟进行分频，该参数拥有16bit的配置能力，故其范围为0x000到0xFFFF；
- CounterMode：即决定定时器是向上计数、向下计数还是中心计数（先上后下）；
- Period：决定定时器的大小，范围从0x000到0xFFFF；
- ClockDivision：首先其不是对于定时器时钟源的分频（Prescaler才是干这个事儿的），其次是下面再说（只有通用和高级定时器有这个功能）；
- RepetitionCounter：依次向上溢出或是向下溢出可以产生一个中断，该参数决定多少次溢出产生一个中断（只有高级定时器有这个功能）。

**Using Timers in Interrupt Mode**
基本定时器不具有外部输出的IO口，所以其只能具有内部计数的功能，而该功能的作用之一就是产生固定频率的中断，到目前为止，基本定时器有以下三个特性

- 计数值通过Periodp配置，其计数范围从0x0000到0xFFFF（16bit有效但在32bit的寄存器中）；
- 计数的频率通过Prescaler配置，其分频范围从0x0000到0xFFFF（16bit）；
- 每当向上或向下溢出，就会产生一个UEV（Update Event），其标志位置高，并且计时器从新从初始值开始计数。

故计时器产生UEV的频率的计算公式为
$$
UpadateEvent = \frac{Timer_{clock}}{(Prescaler + 1) (Period + 1)}
$$
每次定时器中断产生，系统会自动调用`TIM_XXX_IRQHandler()`函数，进而调用`HAL_TIM_IRQHandler(&htim)`，而在HAL层，函数就会判断是定时器的哪一个功能产生的中断，从而清理对应的标志位，再来调用对应功能的`HAL_TIM_XXX_Callback()`函数，然而多次的判断也需要消耗很多的时间。（中断的过程都要经过两个HandlerIRQ和一个Callback函数，其中只有第一个HandlerIRQ不属于HAL层）

对于高级定时器来说，其句柄结构体中有RepetitionCounter参数，即发生多少溢出才产生依次UEV，如此中断频率公式便为
$$
UpadateEvent = \frac{Timer_{clock}}{(Prescaler + 1) (Period + 1) (RepetitionCounter + 1)}
$$
当RepetitionCounter设置为0时，其与基本定时器的中断频率相同。

**How to Choose the Values for Prescaler and Period Fields? **
有空再来写吧，现在先用已有的软件和遍历的算法来计算。

**Using Timers in Polling Mode**
可以通过访问TIMx->CNT寄存器来得到当时定时器的计数值，这也为定时器的阻塞模式提供了基础，只要不断地访问该寄存器直到某一刻的计数值达到了某种要求，即允许其他程序执行下一步，这样对吗？

```c
if (__HAL_TIM_GET_COUNTER(&tim) == value) {
    ...
}
```

其实这样做是不对的，因为定时器是独立于CPU之外的外设，并且CPU访问寄存器来获得计数值需要花费若干周期，所以当定时器的计数频率很大的时候，CPU很有可能会发生跳过某个值的现象发生，所以要用`__HAL_TIM_GET_COUNTER(&tim) >= value`来避免错过某些计数值，或者通过判断UIF标志位，即

```c
if (__HAL_TIM_GET_FLAG(&tim) >= TIM_FLAG_UPDATE) {
    ...
}
```

但是上述两种方法都是基于定时器的溢出发生得时间间隔大于访问计数值得间隔，要是定时器溢出太快也会出问题。

所以综上，定时器是异步得外设，所以请尽量使用中断模式。

**Using Timers in DMA Mode**
若要在固定的时间间隔触发一个事件的发生，常用的方法是使用定时器的中断模式，每隔这段时间就发生一次中断，然后再该中断的回调函数来触发这个事件，如果这个时间涉及到了数据的搬移，那么则可以使用定时器的DMA模式，即定时器每次发生溢出，则会产生一个DMA请求，由此开启一次数据传输。

以下示例为用定时器的DMA请求来触发UART传输数据的示例
```c
int main(void) {
    uint8_t data[] = {0xFF, 0x0};

    HAL_Init();

    Nucleo_BSP_Init();

    htim6.Instance = TIM6;
    htim6.Init.Prescaler = 47999; //48MHz/48000 = 1000Hz
    htim6.Init.Period = 499; //1000HZ / 500 = 2Hz = 0.5s

    __HAL_RCC_TIM6_CLK_ENABLE();

    HAL_TIM_Base_Init(&htim6);

    hdma_tim6_up.Instance = DMA1_Channel3;
    hdma_tim6_up.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_tim6_up.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_tim6_up.Init.MemInc = DMA_MINC_ENABLE;
    hdma_tim6_up.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_tim6_up.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_tim6_up.Init.Mode = DMA_CIRCULAR;
    hdma_tim6_up.Init.Priority = DMA_PRIORITY_LOW;
    HAL_DMA_Init(&hdma_tim6_up);

    HAL_DMA_Start(&hdma_tim6_up, (uint32_t)data, (uint32_t)&GPIOA->ODR, 2);	// 装备DMA，前三步
	HAL_TIM_Base_Start(&htim6);	// 进行第四步的基础
    __HAL_TIM_ENABLE_DMA(&htim6, TIM_DMA_UPDATE);	// 开启DMA，第四步

    while (1);
}
```

首先还是从DMA开启的流程

- 配置定时器和定时器对应DMA的配置；
- 使用`HAL_DMA_Start()`进行确定源地址、目标地址、传输数据大小和装备DMA的操作；
- 开启定时器和定时器的DMA使能，使其定时器能够每次再溢出时产生DMA请求。

**Stopping a Timer**
对应使用以下函数即可

```c
HAL_TIM_Base_Stop();		// 正常计数
HAL_TIM_Base_Stop_IT();		// 中断模式
HAL_TIM_Base_Stop_DMA();	// DMA模式，但是只用来改变Period
```

#### 11.3 General Purpose Timers

通用定时器相比于基本定时器多了很多的功能。

**External Clock Mode 2**
定时器的外部时钟模式2实际上就是简单地切换时钟源，外部（有专门的引脚）或者内部的时钟源，并且该时钟可以进行分频。

此时发生定时器中断的频率计算公式为
$$
UpadateEvent = \frac{Timer_{clock}}{(EXT_{clock}Prescaler)(Prescaler + 1) (Period + 1) (RepetitionCounter + 1)}
$$
其中分母的第一个元素$EXT_{clock}Prescaler$为外部时钟的分频系数。

一个通用定时器的时钟源可用结构体类型`TIM_ClockConfigTypeDef`类型决定，其定义为
```c
typedef struct {
  uint32_t ClockSource;     /*!< TIM clock sources
                                 This parameter can be a value of @ref TIM_Clock_Source */
  uint32_t ClockPolarity;   /*!< TIM clock polarity
                                 This parameter can be a value of @ref TIM_Clock_Polarity */
  uint32_t ClockPrescaler;  /*!< TIM clock prescaler
                                 This parameter can be a value of @ref TIM_Clock_Prescaler */
  uint32_t ClockFilter;     /*!< TIM clock filter
                                 This parameter can be a number between Min_Data = 0x0 and Max_Data = 0xF */
} TIM_ClockConfigTypeDef;
```

- ClockSource表示该定时器时钟的来源，可为Interal或者ETR2（CubeMX中选择），当为Internal时，表示该通用定时器的时钟来源和基本定时器的时钟来源相同，都是内部APB总线的时钟；当为ETR2时，表示该通用定时器开启了External Clock Mode2，即外部时钟模式2，由此可用一个GPIO的引脚来外部输入时钟从而驱动该定时器；
- ClockPolarity：时钟的极性，MX中的选项分别为INVERTED和NONINVERTED，前者为上升沿有效，后者为下降沿有效，原因在于时钟输入默认为低电平，当从低电平到高电平时，发生转换INVERTED，高电平到低电平时，恢复默认NONINVERTED；
- ClockPrescaler：时钟的分频系数，只有当选用ETR2模式的时候才有该选项，即表示对外部时钟的1、2、4、8分频；
- ClockFilter：没懂，一般没使用。

在使用的时候，除了定时器的时钟源不同，其余操作用法与内部源的定时器没什么区别。

**External Clock Mode 1**
当通用定时器被设置为从机模式，则可以使用ETR1的功能，此时MX可以选择多个引脚来进行外部时钟输入，包括内部的ITR0、ITR1、ITR2和ITR3，ETR1引脚，和与通用定时器通道相关的TI1FP1和TI2FP2引脚以及TI1_ED（目前还不清楚这个引脚是什么意思，只知道选择作为外部输入，频率）。当选择了外部时钟输入的引脚后，剩余的做法就和ETR2一样。

通用定时器的从模式可以通过类型`TIM_SlaveConfigTypeDef`来配置从模式，其定义如下
```c
typedef struct {
  uint32_t  SlaveMode;         /*!< Slave mode selection
                                    This parameter can be a value of @ref TIM_Slave_Mode */
  uint32_t  InputTrigger;      /*!< Input Trigger source
                                    This parameter can be a value of @ref TIM_Trigger_Selection */
  uint32_t  TriggerPolarity;   /*!< Input Trigger polarity
                                    This parameter can be a value of @ref TIM_Trigger_Polarity */
  uint32_t  TriggerPrescaler;  /*!< Input trigger prescaler
                                    This parameter can be a value of @ref TIM_Trigger_Prescaler */
  uint32_t  TriggerFilter;     /*!< Input trigger filter
                                    This parameter can be a number between Min_Data = 0x0 and Max_Data = 0xF  */

} TIM_SlaveConfigTypeDef;
```

- SlaveMode：即从模式下的各个子模式，如复位从模式（RESET）、门从模式（GATED）、触发从模式（TRIGGER）和ETR1模式，而本小节中所使用的正是ETR1模式，其工作方式为时钟，具体可看下图
  ![image-20230423210019699](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230423210019699.png)
- InputTrigger：即选择触发的输入ITRI；
- TriggerPolarity：即选择上升沿或者下降沿，在不同的输入引脚会有不同的名称，如ETR1中叫做VERTED和NONVERTED而TI1FP1有Rising edge、Falling edge和Both edge，具体选择就看MX中的配置选项；
- TriggerPrescaler：字面意思，分频；
- TriggerFilter：仍然没有读懂。

所以当使用从模式的ETR1模式时，定时器的溢出时间频率计算公式为
$$
UpadateEvent = \frac{TRGI{clock}}{(Prescaler + 1) (Period + 1) (RepetitionCounter + 1)}
$$
而其中的$TRGI{clock}$已经包含了输入至从模式控制器之前的分频了。

可以看到ETR1和ETR2的效果甚至一些配置选项都几乎相同，都是使用了外部的时钟源来提供给定时器，但是ETR1和ETR2是两种截然不同的方式，ETR2仅仅只把时钟输入改为外部，其他和基本定时器相同，而ETR1模式的本质是使用定时器的从模式，而外部时钟是相对于从模式控制器来说只是触发输入信号。

**Master/Slave Synchronization Modes**
既然我们可以将定时器设置为从模式，那么我们也可以将定时器设置为主模式。此时定时器的作用不再是被动的输入其他来源的触发/时钟信号（TRGI），而是主动的发送触发信号（TRGO），此时可以再设置一个定时器为从模式，让其的触发输入TRGI为前一定时器，则可以完成定时器之间的同步。

定时器的主模式可以通过类型`TIM_MasterConfigTypeDef`来配置，其定义如下
```c
typedef struct {
  uint32_t  MasterOutputTrigger;   /*!< Trigger output (TRGO) selection
                                        This parameter can be a value of @ref TIM_Master_Mode_Selection */
  uint32_t  MasterSlaveMode;       /*!< Master/slave mode selection
                                        This parameter can be a value of @ref TIM_Master_Slave_Mode
                                        @note When the Master/slave mode is enabled, the effect of
                                        an event on the trigger input (TRGI) is delayed to allow a
                                        perfect synchronization between the current timer and its
                                        slaves (through TRGO). It is not mandatory in case of timer
                                        synchronization mode. */
} TIM_MasterConfigTypeDef;
```

- MasterOutputTrigger：确定TRGO的输出形式，可看一下表格
  ![image-20230423222956760](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230423222956760.png)
  ![image-20230423223010108](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230423223010108.png)
  - 其中现在已经了解的有`ENABLE`，即当主计时器打开，就会发送一个TRGO，常常用来同步开启所有定时器；
  - `UPDATE`，即主定时器发生溢出事件，则产生一个TRGO，常用来将主定时器作为分频器；
- MasterSlaveMode：没看懂，什么TRGO会同步什么的，反正主从模式都设置的时候就要打开。

以下这个例子使用了两个通用定时器TIM1和TIM3，将TIM1的时钟模式设置为ETR2模式，由此其可以接收外部的时钟，TIM1的从模式设置为rising trigger，即上升沿触发，并开通TI1FP1（记得再GPIO设置的TIM选项中对应的IO口设置为下拉），由此每次TI1FP1的第一次上升沿会开启TIM1的时钟计时，从而开启接下来之后的定时器计时，TIM1的主模式设置为Update Event，由此每当其发生定时器溢出，就可产生输出触发信号TRGO；将TIM3的从模式设置为ETR1，由此其可将TIM1的TRGO视为自己的输入时钟，由此开始计时。上述实现的功能可简述为，TIM1的TI1FP1上升沿触发所有定时器开始定时，而 TIM1作为TIM3的分频器定时，从而完成不同定时器之间的同步功能。
在MX配置完成之后，主代码的编写及其简单，只要简单的使用`HAL_TIM_Base_Start_IT(&htim1)`和`HAL_TIM_Base_Start_IT(&htim3)`来开启定时器即可，再到中断回调函数中编写对应代码。

```c
int main_cpp() {
    RETARGET_Init(&huart4);

    HAL_TIM_Base_Start_IT(&htim1);
    HAL_TIM_Base_Start_IT(&htim3);
    while (true) {

    }
    return 0;
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM1) {
        printf("tim1 upadate\n");
    }
    if (htim->Instance == TIM3) {
        printf("tim3 upadate\n");
    }
}
```

**Generate Timer-Related Events by Software**
定时器的事件不仅可以由硬件产生，如定时器的计数器溢出产生UEV，也可以通过软件产生，能够通过该方式产生事件的原因在于STM32中专门有寄存器EGR来支持软件产生定时器事件，EGR中的每一个位对应一个事件，如第一位就是对应该定时器的UEV事件。

HAL库支持使用函数

```c
HAL_StatusTypeDef HAL_TIM_GenerateEvent(TIM_HandleTypeDef *htim, uint32_t EventSource);
```

来软件方式产生定时器事件，其中第二个参数`EventSource`可选择以下表的参数

![image-20230425212611227](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230425212611227.png)

- TIM_EVENTSOURCE_UPDATE：就是产生该定时器的UEV，这里只能通过软件设置；

  - 这一位在该定时器为主机模式且TRGO设置为Reset时特别有用，该定时器产生触发输出的条件就是软件产生UEV事件。
    例如这里将TIM1设置为主机模式，TRGO设置为Reset，此时将程序写成每隔一段时间产生触发信号，而TIM3使用ETR1模式即将该触发信号设置为TIM3的时钟源

    ```c
    int main_cpp() {
        RETARGET_Init(&huart4);
        
        HAL_TIM_Base_Start(&htim1);
        HAL_TIM_Base_Start_IT(&htim3);
        while (true) {
            HAL_TIM_GenerateEvent(&htim1, TIM_EVENTSOURCE_UPDATE);
            HAL_Delay(500);
        }
        return 0;
    }
    ```

    注意不要在TIM1的中断中使用这个函数产生UEV事件，因为此时会再次引出下一个中断，导致程序一直在中断函数中；

  - 什么ARR，什么影子寄存器没学懂。

- 其他的没学懂。

**Input Capture Mode**
通用和高级定时器还支持很多除了计数之外的功能，从他们的框图中可以看出
![image-20230425220234462](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230425220234462.png)

每一个外部的输入通过TIx的选择器连到边缘检测器上，同时其还能进行滤波来防止抖动（这里之前都属于通道的常规操作），其后再连到ICx的选择器来（可能）进行"remap"操作（这里之后都是），这是为了防止当前的通道IO口被占用从而转到其他的通道IO，但是还是用本通道的部分来进行定时器的功能。

输入捕获模式可以用来计算外部信号的频率，外部信号可以连接到某一定时器的四个通道，且这四个通道相互之间是独立的。

假设外部输入一个方波，并且其连接到某一开启输入捕获的通道，则该通道会通过边缘检测器，用TIMx_CCR寄存器来记录当时的CNT，下图展示了这一过程
![image-20230425221533443](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230425221533443.png)

我们设置了上升沿检测的输入捕获，所以每次上升沿就会使用TIMx_CCRx来记录当前的CNT值，如上图记录了4和20，然后通过这两个值来计算该方波的频率，计算公式如下
$$
Period = Capture \cdot (\frac{TIMx\_CLK}{(Prescaler + 1)(CH_{Prescaler})(Polarity_{Index}))})^{-1}
$$
其中，$Prescaler$表示定时器时钟的分频，$Capture$表示相邻边缘的计数值增量，计算公式见书P302；$CH_{Prescaler}$表示的定时器通道的分频；$Polarity_{Index}$表示单上升沿或单下降沿捕获（此时为1）还是上升沿和下降沿都捕获（此时为2）。

还有一点需要注意，我们需要将UEV的频率尽可能的变慢，因为如果UEV快的话，有可能出现外部信号的周期过长从而导致相邻边沿检测中间夹着着很多次的UEV，从而导致不准，所以外部信号的频率要快于UEV的频率，减慢UEV频率的方式就是使用尽可能大的定时器分频（尽量大的分频会导致测量的精度下降）和尽量大的计数器范围。

输入捕获测得的外部方波信号频率范围和精度。

1. 最高频率，输入捕获的最高频率由计时器计数的频率所约束的，所能测得的最高频率就是对应着`diff_CNT`的值为1，此时最高频率公式如下
   $$
   f_{measure\_MAX} = \frac{TIMx\_CLK}{Prescaler + 1}
   $$

2. 最低频率，其由UEV的频率所限制，具体原因前面已经说明，最低频率公式如下
   $$
   f_{measure\_MIN} = \frac{TIMx\_CLK}{(Prescaler + 1) (Period + 1)}
   $$

3. 精度，通过推导，可以得到所测频率的公式如下
   $$
   f_{measure} = \frac{TIMx\_CLK}{(Capture)(Prescaler + 1)}
   $$
   可以看到，是经典的反比例函数形式，分母值越大，则精度越高，分母值越小，则精度越低，具体还得学数论？

定时器的输入捕获模式通过结构体`HAL_TIM_InitTypeDef`类型来配置，定义如下
```c
typedef struct {
  uint32_t  ICPolarity;  /*!< Specifies the active edge of the input signal.
                              This parameter can be a value of @ref TIM_Input_Capture_Polarity */

  uint32_t ICSelection;  /*!< Specifies the input.
                              This parameter can be a value of @ref TIM_Input_Capture_Selection */

  uint32_t ICPrescaler;  /*!< Specifies the Input Capture Prescaler.
                              This parameter can be a value of @ref TIM_Input_Capture_Prescaler */

  uint32_t ICFilter;     /*!< Specifies the input capture filter.
                              This parameter can be a number between Min_Data = 0x0 and Max_Data = 0xF */
} TIM_IC_InitTypeDef;
```

- ICPolarity：捕获外部信号的上升沿、下降沿和上升下降沿；
- ICSelection：选择进入ICx的信号（上图可见），有以下值可以选择，一般选择第一个选项，其他选项用于支持“remap”操作，具体还是不懂；
  ![image-20230505215845897](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230505215845897.png)
- ICPrescaler：对要进行捕获的输入信号进行分频；
- ICFilter：和其他功能一样的滤波器。

以下为使用输入捕获测量外部方波频率的例子
```c
volatile uint8_t capture_done = 0;
int main_cpp() {
    RETARGET_Init(&huart4);

    uint16_t captures[2] = {0};
    HAL_TIM_IC_Start_DMA(&htim4, TIM_CHANNEL_1, (uint32_t *)captures, 2);
    while (true) {
        if (capture_done == 1) {
            capture_done = 0;
            uint16_t diff_capture = captures[0] < captures[1] ? 
                					captures[1] - captures[0] :
                    				htim4.Instance->ARR - captures[0] + captures[1];

            double freq = (double)HAL_RCC_GetPCLK1Freq() * 2 / 
                		  (htim4.Instance->PSC + 1) / 
                		  (double)diff_capture;
            printf("%f\n", freq);
            HAL_TIM_IC_Start_DMA(&htim4, TIM_CHANNEL_1, (uint32_t *)captures, 2);
            HAL_Delay(500);
        }
    }
    return 0;
}

void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM4) {
        capture_done = 1;
    }
}
```

首先复习以下DMA的使用，从DMA的角度来说，需要进行四步（具体看前面DMA章节），前三步可用`HAL_DMA_Start()`函数来进行，最后一步则是对应外设的DMA开启；从外设的角度来说，HAL库大部分可以直接使用一个函数来进行所有步骤的配置，如这里定时器输入捕获的`HAL_TIM_IC_Start_DMA()`函数。

首先是定义了`captures`的数组变量，该变量用于保存DMA从外设定时器计数器搬运到内存的数据，接着直接使用`HAL_TIM_IC_Start_DMA()`函数来获得该数组的数据，由于使用DMA不能马上获得数据，所以还需要一个全局变量来标识DMA的完成，也就是再输入捕获的中断函数中改变该标识的量，如这里定义了
```c
volatile uint8_t capture_done = 0;
```

其中`volatile`告诉编译器用该关键字的变量是易改变的，通常用于设置为状态标志变量。

函数`HAL_RCC_GetPCLK1Freq`用来获取PCLK1的时钟频率，但是用于定时器的时钟频率为通常为PCLKx的两倍（具体看MX的时钟树），所以这里还需要乘2。

**Output Compare Mode**
目前我们已经知道了定时器可以通过计数器溢出来产生事件，从而触发中断或者DMA，通用或高级定时器还支持输出比较模式，即将某一值Pulse存入CCRx寄存器（注意，输入捕获中存储CNT的值的寄存器也是CCRx），只要当计数值CNT与CCRx中的值匹配（相等）时，就会产生一事件。

输出比较模式通过结构体`TIM_OC_InitTypeDef`来初始化某一定时器输出比较模式的配置，其定义如下
```c
typedef struct {
  uint32_t OCMode;        /*!< Specifies the TIM mode.
                               This parameter can be a value of @ref TIM_Output_Compare_and_PWM_modes */

  uint32_t Pulse;         /*!< Specifies the pulse value to be loaded into the Capture Compare Register.
                               This parameter can be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF */

  uint32_t OCPolarity;    /*!< Specifies the output polarity.
                               This parameter can be a value of @ref TIM_Output_Compare_Polarity */

  uint32_t OCNPolarity;   /*!< Specifies the complementary output polarity.
                               This parameter can be a value of @ref TIM_Output_Compare_N_Polarity
                               @note This parameter is valid only for timer instances supporting break feature. */

  uint32_t OCFastMode;    /*!< Specifies the Fast mode state.
                               This parameter can be a value of @ref TIM_Output_Fast_State
                               @note This parameter is valid only in PWM1 and PWM2 mode. */


  uint32_t OCIdleState;   /*!< Specifies the TIM Output Compare pin state during Idle state.
                               This parameter can be a value of @ref TIM_Output_Compare_Idle_State
                               @note This parameter is valid only for timer instances supporting break feature. */

  uint32_t OCNIdleState;  /*!< Specifies the TIM Output Compare pin state during Idle state.
                               This parameter can be a value of @ref TIM_Output_Compare_N_Idle_State
                               @note This parameter is valid only for timer instances supporting break feature. */
} TIM_OC_InitTypeDef;
```

- OCMode：输出比较模式的子模式选择，可以选择以下表格的模式
  ![image-20230426141859650](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230426141859650.png)
  - TIMING：就是正常的基本定时器作用，不涉及CCRx寄存器，MX中称为frozen模式；
  - ACTIVE：CNT和CCRx相匹配时就置高输出电平；
  - INACTIVE：CNT和CCRx相匹配时就置低输出电平；
  - TOGGLE：CNT和CCRx相匹配时就转换输出电平；
  - FORCED_ACTIVE：强制置高，无论CNT为多少；
  - FORCED_INACTIVE：强制置低，无论CNT为多少。
- Pulse：就是待存入CCRx寄存器中要与CNT比较的值；
- 其他的不会。

以下的示例为使用输出比较模式输出同一定时器的两个通道的方波
```c++
int main_cpp() {
    RETARGET_Init(&huart4);

    HAL_TIM_OC_Start_IT(&htim2, TIM_CHANNEL_1);
    HAL_TIM_OC_Start_IT(&htim2, TIM_CHANNEL_4);
    while (true) {

    }
    return 0;
}

// 每次中断都会设置对应的htim->Channel
void HAL_TIM_OC_DelayElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM2) {
        uint32_t pulse;
        uint32_t arr = __HAL_TIM_GET_AUTORELOAD(htim);  // 获得arr的值

        if (htim->Channel == HAL_TIM_ACTIVE_CHANNEL_1) {
            pulse = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1); // 获得当前的pulse值
            if (pulse + 1000 < arr) {
                __HAL_TIM_SET_COMPARE(htim, TIM_CHANNEL_1, pulse + 1000);
            }
            else {
                __HAL_TIM_SET_COMPARE(htim, TIM_CHANNEL_1, pulse + 1000 - arr);
            }
        }

        if (htim->Channel == HAL_TIM_ACTIVE_CHANNEL_4) {
            pulse = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_4); // 获得当前的pulse值
            if (pulse + 1000 < arr) {
                __HAL_TIM_SET_COMPARE(htim, TIM_CHANNEL_4, pulse + 1000);
            }
            else {
                __HAL_TIM_SET_COMPARE(htim, TIM_CHANNEL_4, pulse + 1000 - arr);
            }
        }

    }
}
```

- 进行基本定时器的配置，即配置Prescaler和Period；
- 打开需要的通道，如本例中的Channel_1和Channel_2；
- 只需要配置OC的Mode和Pulse即可，其他的默认（以后再来查有什么用）；
- 最后写下代码，主函数中只需要开启中断模式的`HAL_TIM_OC_Start_IT()`函数；
- 而需要在中断中不断地改变Pulse的值，才能满足方波频率的生成，即每次中断，将Pulse提升一个步进，此时初始的Pulse就是对应的初相，理论上不同通道的有一个微小的相移，所以实际调节初相还需要进行微调。
  - `__HAL_TIM_GET_AUTORELOAD()`宏定义的函数，用于得到当前定时器的Period的值；
  - 注意中断回调中的`htim->Channel`是会不断调整的`HAL_TIM_ActiveChannel`类型变量，代表着引起中断的该定时器的相应活跃的通道，所以中断的判断不能使用`TIM_Channel_x`，而是使用`HAL_TIM_ACTIVE_CHANNEL_x`；
  - `HAL_TIM_ReadCapturedValue()`函数用来获取当前定时器的CRRx的值，前面说过，OC和IC的CRRx在同一通道是公用的（因为同一通道不可能即设置为IC模式由设置为OC模式），所以该函数的名字为`ReadCaptureValue`；
  - `__HAL_TIM_SET_COMPARE()`的宏定义就是设置OC模式的Pulse的值。

上述程序也可改为DMA模式，首先现在MX中的对应TIM设置中开启DMA，DMA的方向一定要注意（这里是内存到外设）；由于Pulse宽32bit，所以数据大小改为32bit；由于需要一直传输数据，所以DMA模式改为循环模式。
可以使用以下函数来进行OC的DMA模式

```c
HAL_StatusTypeDef HAL_TIM_OC_Start_DMA(TIM_HandleTypeDef *htim, uint32_t Channel, 
                                       		uint32_t *pData, uint16_t Length);
```

OC的DMA传输最终改变的是该定时器通道的CCRx寄存器，并且每次CNT与CCRx匹配就会产生一个DMA请求，从而改变CCRx，由此其的功能是和上述的中断模式完全相同的，就是生成Pulse的数组比较麻烦，最终得到代码如下
```c
// 定时器输出比较
int main_cpp() {
    RETARGET_Init(&huart4);

    uint32_t pulses[10];	// 预先定义数组，数组内的元素就是要改变的Pulse的值。
    uint16_t arr_tim2 = __HAL_TIM_GET_AUTORELOAD(&htim2);
    int i = 0;
    // 这里一定把Pulse设置为Period的因数
    for (uint32_t pulse = 0; pulse < arr_tim2; pulse += 1000) {
        pulses[i++] = pulse;
    }

    HAL_TIM_OC_Start_DMA(&htim2, TIM_CHANNEL_1, (uint32_t *)pulses, 10);
    HAL_TIM_OC_Start_DMA(&htim2, TIM_CHANNEL_4, (uint32_t *)pulses, 10);
    while (true) {

    }
    return 0;
}
```

**Pulse-Width Generation**
PWM的概念在这里不多赘述，也就是脉宽调制，PWM模式同样通过定时器输出比较模式的结构体`TIM_OC_InitTypeDef`来配置，与输出比较模式不同的是，有专门的初始化函数`HAL_TIM_PWM_ConfigChannel()`来接受该结构体从而进行PWM模式的初始化

- OCMode：在PWM模式中可选择的选项有PWM mode1和PWM mode2，前者在定时器计数器值小于Pulse的值时为高电平，后者相反；
- Pulse：即确定PWM波的脉宽，其必须在0至Period之间。

以下是采用PWM模式来实现呼吸灯效果的程序
```c++
int main_cpp() {
    RETARGET_Init(&huart4);
    // Start your coding
    HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_2);
    uint32_t pulse = 0;
    while (true) {
        while (pulse < __HAL_TIM_GET_AUTORELOAD(&htim4)) {
            ++pulse;
            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_2, pulse);
            HAL_Delay(1);
        }
        while (pulse > 0) {
            __HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_2, pulse);
            --pulse;
            HAL_Delay(1);
        }
    }
    return 0;
}
```

首先是该定时器的基本配置，在MX中完成，这里不赘诉；在代码中，首先打开了PWM模式，接着在循环中使用`__HAL_TIM_GET_AUTORELOAD()`的宏定义函数来获取当前定时器的Period值，最后使用`__HAL_TIM_SET_COMPARE()`函数来改变寄存器CCRx中的值，从而设置PWM的不同脉宽。

PWM波加外围电路产生其他波形的没有看，下次再看。

**One Pulse Mode**

#### 11.4 SysTick Timer



### 12. Analog-To-Digital Conversion

#### 12.1 Introduce to SAR ADC

![image-20230616152130107](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230616152130107.png)

如上图为STM32简化的ADC结构图，其中Input selection & scan control单元用于选择ADC的输入源。取决于不同的ADC模式，有多种不同的输入源可供选择，以此完成单次、扫描和连续模式的功能；Start & Stop Control单元用于开启或定时AD转换的过程，其可以软件或其他中断源来触发（特别的，定时器触发从而实现固定采样率采样）。

![image-20230616153023822](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230616153023822.png)

如上图为SAR ADC的内部结构图，其中包含一个SHA（Sample-and-Hold）模块。总所周知，ADC的转换过程是需要时间的，在转换的过程中必须得保持ADC的输入电压不变，而SHA就是用来完成这一功能的模块。

**How Successive Approximation Register ADC Works**
首先来说一个简单的前置知识，若你已知一个连续的双闭区间范围为$[a, b]$，那么你该怎么找到这个范围的中心呢？很显然，对于连续来说，范围的中心是$(a + b) / 2$，而中心距离两边的距离为$(b - a) / 2$。

若$a$和$b$都是整数，且让这个范围变得离散，即只能取得整数的值，那么情况就会略微复杂，我们假设这个区间（或者说是序列）的个数为$N$，那么很显然，$N = b - a + 1$，依然按照连续的方法来求得该区间的中心，也即$(a + b) / 2$，将$N$的表达式代入而取代$a$或$b$，则得到$(a + b) / 2 = a + (N - 1) / 2 = b - (N - 1) / 2$。

我们首先讨论更为简单的$N$为奇数的情况，因为我们知道，奇数序列的区间中间值是唯一的，例如$\{3, 1, 2\}$的中间值为2。此时我们使用刚才的式子来计算中间值，可以得到$a + (N - 1) / 2$为整数，其完全能够对应我们刚才的结论，即奇数个的序列只有一个中间值。

最扰人的结果出现在$N$为偶数的情况，此时出现的中间值有两个，例如$\{3, 1, 2, 4\}$的中间值可以认为是1或2。我们同样带入上式可以得到结果$(a + b) / 2 = a + (N - 1) / 2 = b - (N - 1) / 2$的值此时不为整数，此也意味着真正的中心在两个中间值的中间。处于计算机计算的背景条件，由于计算机进行整数的除法运算都是向下取整，所以依据此公式计算出的奇数个的序列的中值即为靠前的中间值。在实际的情况中，常出现的序列的起始位置$a = 0$，也即范围为$[0, N - 1]$，如此我们通过公式得到的中间位置为$(N - 1) / 2$，刚好就是这个区间最大值的一半。

我们总结一下序列计算中间位置的公式

- 知区间范围为$[a, b]$，则直接$(a + b) / 2$；
- 知区间的起始位置$a$和序列长度$N$，则直接$a + (N - 1) / 2$；
- 知区间的起始位置为0和序列长度$N$，则直接区间最大位置的一半$(N - 1) / 2$；
- 以上全基于向下取整的环境。牢记第三点，**起始位置为0，那么最大位置的一半就是中间偏前的位置，序列长度的一半就是中间偏后的位置**。

看一个应用例子，假设8位二进制数，快速计算其的中间值。
显然，8位二进制数以0开始，最大数为${1111\ 1111}_{(2)}$，那么中间值且偏前的数就是整体右移为${0111 \ 1111}_{(2)}$，中间值偏后就是${1000 \ 0000}_{(2)}$。虽然以上所讨论的十分简单，但是我觉得这是理解SAR ADC算法步骤的关键。

SAR ADC的原理本质为二分查找，所以为了复习二分查找，请参考leetcode刷题笔记。二分查找可以视为由两个区域向中间扩张从而卡出边界的问题。在SAR ADC中，左边区域的条件为$V_{IN} \leq V_{DAC}$，而右边区域为$V_{IN} > V_{DAC}$（因为理论上$V_{IN}$和$V_{DAC}$永远不会相等，所以等号在左边或右边区域无所谓）。由此便有了一下的步骤

1. SAR ADC的寄存器（也即DAC的寄存器）首先将所有位清位0，此时所有位都未进行过更改；
2. 将所有未更改的位中的最高位置为1；
3. 由此时的DAC输出$V_{DAC}$和$V_{IN}$进行比较，若$V_{IN} \leq V_{DAC}$，则将此位置为0；
4. 若$V_{IN} > V_{DAC}$，则将此位置为1；
5. 重复使得从步骤2开始。

上述步骤如何体现出区域扩张的思想呢？首先在上述的区域扩张中，我们认为每次取中间值的过程即为一次区域扩张的过程，要么是左边区域扩张，要么是右边区域扩张，即每次循环都有`left = mid`（`left`增大）或`right = mid`（`right`减小），由此最终卡出边界。那么SAR ADC是否有这样的过程呢？我们认为步骤2，即置1的过程就是在取中间值，当$V_{DAC} < V_{IN}$时，应该是左边区域的扩张（较小值的区域），此时我们将刚刚置的1保持不变，移步到下一位，若我们此时没轮上的位都为0，那么该数此时的整体数值就是增大（相比于取中值之前），所以刚好对应上了左边边界的扩张；当$V_{DAC} > V_{IN}$时，我们将刚置的1恢复为0，同样为轮上的位为0，那么此时该数数值不变，同样对应着右边扩张时左边边界的不变，若把没轮上的位视为1，则此时整体数值减小，对应着右边边界的减小。综上，对于ADC中的一次循环，将轮过的位保持不变，则未轮上的部分置0对应着左边边界，置1对应右边边界，所以对于SAR ADC，其一个数值就同时反应了左边边界和右边边界。
也可以简单理解为，置1取中值导致了整体数值的增大，就像是`left = (left + right) / 2`，即左边边界的增大。

还有一个问题关于边界，区域扩张的理论假定左右边界的初始值分别是-1和总区域长度，由此不会造成超出区域的漏判问题，因为此时我们要求目标数超出区域时也能检测出来。但是SAR ADC的初始值就是简单的0和最大值，因为ADC不要求检测出是否超出范围，超范围时用最近的边界表示就可以了。

#### 12.2 `HAL_ADC` Module

**12.2.4 A/D Conversions in Polling or Interrupt Mode**
启动AD阻塞模式的大致流程为：首先开启ADC，此时ADC会不断的进行采集和转换；读取数据，直接从寄存器中读取。
HAL库中配合函数来实现上述流程，如下

```c++
int main_cpp() {
    RETARGET_Init(&huart1);
    // Start your coding
    auto raw_data = uint16_t(0);
    auto temp = float(0.0);
    
    HAL_ADC_Start(&hadc3);		// 开启ADC
    while (true) {	// 由于使用连续模式，在停止之前ADC会不断进行采集
        HAL_ADC_PollForConversion(&hadc3, HAL_MAX_DELAY);	// 这步不用也可以，因为转换一次过程太快了

        raw_data = HAL_ADC_GetValue(&hadc3);
        temp = ((float)raw_data) / 65535 * 3300;

        printf("ADC rawdata: %hu\r\n", raw_data);
        printf("Voltage: %f\r\b\r\n", temp);
        HAL_Delay(500);
    }
    return 0;
}
```

上述代码使用了STM32的单通道连续转换模式，即在CUBEMX中关闭扫描模式（即打开单通道模式），打开连续模式（即开始后没遇到停止就会一直转换）。作为区分，单通道单次转换模式仍然能够实现上面代码的效果，但是每次循环都要先开启ADC，循环最后关闭ADC（使用`HAL_ADC_Stop()`函数）。

由于ADC在连续模式下的转换速度太快了，所以就算不使用`HAL_ADC_PollForConversion()`函数，效果也差不多。

注意：在H7中，有MX选项为Overrun behaviour，在使用连续转换模式的时候一定要选择Overrun data overwritten，即每次转换一个新数据，就把之前数据给覆盖掉，若不覆盖，EOC标志永远不会开启，则连续模式会一直卡在`HAL_ADC_PollForConversion()`函数中。

中断模式的配置和阻塞模式一模一样，唯一区别就是中断模式需要打开中断使能，中断模式的代码如下
```c++
int main_cpp() {
    RETARGET_Init(&huart1);

    HAL_ADC_Start_IT(&hadc3);
    while (true) {

    }
    return 0;
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
    uint16_t raw_data;
    auto temp = float(0.0);

    raw_data = HAL_ADC_GetValue(&hadc3);
    temp = ((float)raw_data) / 65535 * 3300;

    printf("ADC rawdata: %hu\r\n", raw_data);
    printf("Voltage: %f\r\b\r\n", temp);
}
```

