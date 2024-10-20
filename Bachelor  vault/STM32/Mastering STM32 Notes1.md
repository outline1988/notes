###  6. GPIO Management

STM32原有官方的标准库（Standard Peripheral Library），但是由于STM32众多的产品，不能保证所有标准库始终保持一致，并且还有很多的bug，所以出现了HAL（Hardware Abstraction Layer）库。

#### 6.1 STM32 Peripherals Mapping

![image-20220924111718434](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20220924111718434.png)

总线矩阵（Bus Matrix）由两个主机（Master）Core和DMA轮番控制，总线分别于四个从机（Slave）相连，分别是与Flash memory相连的Flash interface、SRAM、通过AHB2（Advanced High-performance Bus）相连的GPIO和与AHB1相连的bridge，并且其通过APB bus（Advanced Peripheral Bus）与各种外设相连。

DMA有着与其他所有从机或外设相连的通道（控制着外设和SRAM的通道）。

#### 6.2 GPIO Configuration

**传统方法**
用`pointer to uint32_t`类型来表示一个32位寄存器，并给它传入寄存器所处的地址。使用`|`或操作来设置某个位。

**HAL库方法**：
使用GPIO_TypeDef（其就是一个C语言的struct类型）来代表某个GPIOx。

HAL中使用指向GPIO_TypeDef的指针来表示一个GPIOx，其类型定义如下
```c
typedef struct {
  __IO uint32_t MODER;    /*!< GPIO port mode register,               Address offset: 0x00      */
  __IO uint32_t OTYPER;   /*!< GPIO port output type register,        Address offset: 0x04      */
  __IO uint32_t OSPEEDR;  /*!< GPIO port output speed register,       Address offset: 0x08      */
  __IO uint32_t PUPDR;    /*!< GPIO port pull-up/pull-down register,  Address offset: 0x0C      */
  __IO uint32_t IDR;      /*!< GPIO port input data register,         Address offset: 0x10      */
  __IO uint32_t ODR;      /*!< GPIO port output data register,        Address offset: 0x14      */
  __IO uint32_t BSRR;     /*!< GPIO port bit set/reset register,      Address offset: 0x18      */
  __IO uint32_t LCKR;     /*!< GPIO port configuration lock register, Address offset: 0x1C      */
  __IO uint32_t AFR[2];   /*!< GPIO alternate function registers,     Address offset: 0x20-0x24 */
} GPIO_TypeDef;
```

有关GPIOx的寄存器都是连续的一段内存，所以可以使用struct结构中成员的位置来表示各个寄存器的相对于地址的偏移。GPIO_TypeDef都已经宏定义好了，也即是说，一个GPIO_TypeDef的指针就实际代表一个GPIOx（注：这里的地址都是字节，表示8位）

HAL库使用GPIO_InitTypeDef类型的结构来进行初始化，其定义如下
```c
typedef struct {
  uint32_t Pin;       
  uint32_t Mode;      
  uint32_t Pull;      
  uint32_t Speed;   
  uint32_t Alternate;
} GPIO_InitTypeDef;
```

- PIN：就是GPIOx中16个口的其中之一，可以通过或操作实现同时定义PIN，如`GPIO_PIN_0 | GPIO_PIN_1`，之所以能用该方式进行，是因为每个GPIO_PIN_N是一个16位中第N位为1，其余为0的uin16_t类型变量，如`GPIO_PIN_9`为`((uint16_t)0x0200)`。
- Mode：对应GPIO口的12仲模式，如推挽，开漏等等，这里不详细介绍。
- Pull：上拉下拉或不上拉也不下拉。
- Speed：速度，详细后面来谈。
- Alternate：详细后面来谈。

这个类型就是为了方便将某一个GPIOx（包括16个pin）中的各种参数传入具体的GPIOx指针中（GPIO_TypeDef）而设立的。

在初始化函数中，通过新定义的自动变量，并且传址的方式来传入该变量，从而实现GPIOx的初始化，此过程是依靠下面的函数实现的。

```c
void HAL_GPIO_Init(GPIO_TypeDef  *GPIOx, GPIO_InitTypeDef *GPIO_Init);
```

该函数传入GPIOx对应的地址用来代表GPIOx，对应的地址宏已经写好，用GPIO_Init来表示初始化的内容。

综上所述，GPIOx的配置是以GPIOx为大主体，然后其中的Pin为小主体来配置的。首先通过对GPIO_InitTypeDef类型来配置对应GPIOx的Pin和其他参数，然后再调用HAL_GPIO_Init函数来初始化GPIO_TypeDef指针类型的GPIOx。

**The process of GPIO initialization in HAL**
MX将所有的GPIOx当成一个外设，并将其的初始化定义在`MX_GPIO_Init()`函数中，该函数首先以GPIOx为一个大模块，Pin为小模块，分别将GPIO_InitTypeDef类型的GPIO_InitStruct结构成员赋值，并传入`HAL_GPIO_Init()`函数来初始化，可以重复调用该函数来完成不同功能的Pin设置。

其他的外设都是将表示外设的地址和外设的初始化参数再次封装成一个handler结构，由此初始化函数只需要有一个handler对应的参数。但是GPIO之所以不这么干的原因，我猜测为GPIO不像其他的外设一样，每一个的GPIOx都有很多的Pin，对应着很多不同的功能，所以如果要使用handler封装，则需要对应的很多的handler变量，即每一个GPIOx对应一个hander类型；同时，GPIO口必须得支持不断更新的特性，所以不能将一整块GPIO当作一个变量进行。

总的来说，就是GPIO作为一个整体太过于庞大和复杂，所以不能将其整个封装成一个handler，加之其要支持可以更新的特性，所以必须以小的单位Pin来进行初始化（为了更精简代码，当有几个Pin的功能等初始化参数完全相同，可以自动以一个初始化函数来初始化多个Pin）。

```c
// gpio.c
void MX_GPIO_Init(void)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  // ...

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOE, KEY_0_Pin|KEY_1_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_1|DSS_FS_Pin|DSS_PS_Pin|DDS_FSY_Pin
                          |DDS_SCK_Pin|DDS_RST_Pin|DDS_SDK_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, OLED_DC_Pin|OLED_RST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pins : PEPin PEPin */
  GPIO_InitStruct.Pin = KEY_0_Pin|KEY_1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pin : PtPin */
  GPIO_InitStruct.Pin = KEY_UP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(KEY_UP_GPIO_Port, &GPIO_InitStruct);

  // ...

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI0_IRQn);
  // ...

}
```

函数`MX_GPIO_Init()`进行对GPIO的初始化

- 先将对应GPIOx的时钟打开。
- 配置Output Level，就是初始电平为高还是低。
- 配置GPIO_InitStruct，包含模式、上下拉（对于输出模式影响不大）和输出速度。
- 若有GPIO设置为中断模式，最有配置中断。

**GPIO_Mode**
![f2cd060c628a451890f6471b0500ba56](https://img-blog.csdnimg.cn/f2cd060c628a451890f6471b0500ba56.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5pif6L6w77-h5aSn5rW3,size_20,color_FFFFFF,t_70,g_se,x_16)

```c
#define  GPIO_MODE_INPUT                        MODE_INPUT                                                 
#define  GPIO_MODE_OUTPUT_PP                    (MODE_OUTPUT | OUTPUT_PP)                                   
#define  GPIO_MODE_OUTPUT_OD                    (MODE_OUTPUT | OUTPUT_OD)                                   
#define  GPIO_MODE_AF_PP                        (MODE_AF | OUTPUT_PP)                                       
#define  GPIO_MODE_AF_OD                        (MODE_AF | OUTPUT_OD)                                       

#define  GPIO_MODE_ANALOG                       MODE_ANALOG                                                 
    
#define  GPIO_MODE_IT_RISING                    (MODE_INPUT | EXTI_IT | TRIGGER_RISING)                     
#define  GPIO_MODE_IT_FALLING                   (MODE_INPUT | EXTI_IT | TRIGGER_FALLING)                   
#define  GPIO_MODE_IT_RISING_FALLING            (MODE_INPUT | EXTI_IT | TRIGGER_RISING | TRIGGER_FALLING)   
 
#define  GPIO_MODE_EVT_RISING                   (MODE_INPUT | EXTI_EVT | TRIGGER_RISING)                     
#define  GPIO_MODE_EVT_FALLING                  (MODE_INPUT | EXTI_EVT | TRIGGER_FALLING)                   
#define  GPIO_MODE_EVT_RISING_FALLING           (MODE_INPUT | EXTI_EVT | TRIGGER_RISING | TRIGGER_FALLING)   
```

`GPIO_MODE_EVT_x`用于睡眠模式。

**GPIO Alternate Function**
GPIO口的资源除了可以用于配置GPIO的各种模式，同样也用于内部的各种外设，可以通过数据手册或仅仅通过MX来查看各种GPIO口支持的内部外设。

使用内部的外设都是基于正确的GPIO口配置，并且系统也会在你选择的某个内部外设的功能上自动帮你配置好，在此基础上在封装一层库从而实现特定的内部外设功能。

**Understanding GPIO Speed**
该参数只能在output模式下有效，并且不是代表GPIO口的交替速率，而是从低电平到高电平上升沿的时间，称为slew speed，并且不同的产品其绝对速率不一样（甚至连HAL库的名字都不一样...）。不过在没有特殊要求的情况下还是保持最小的速率，以减少其他资源的消耗。

ST规定的最大交换速率为两个周期，也就是AHB频率的一半，但实际使用的时候还会有额外很多的开销，所以一半会小个四五倍。（猜的，大概，要用的时候再来测）

#### 6.3 Driving a GPIO

四个函数可以用来操作GPIO，这四个函数都是通过GPIOx口的地址（由GPIO_TypeDef指针来决定）和Pin来表示。
```c
void HAL_GPIO_WritePin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin, GPIO_PinState PinState);
GPIO_PinState HAL_GPIO_ReadPin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);
void HAL_GPIO_TogglePin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);
HAL_StatusTypeDef HAL_GPIO_LockPin(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin);
```

最后一个函数将GPIO口锁住，直到重置。

可以使用以下函数来将某个GPIO口设置为默认格式
```c
void HAL_GPIO_DeInit(GPIO_TypeDef *GPIOx, uint32_t GPIO_Pin);
```

通常在不再需要使用某个外设，或者睡眠模式的时候使用来减小功耗。

### 7. Interrupts Management

中断（interrupt）是导致当前代码停止转而到更高优先级程序的异步事件，用来管理这一过程的代码叫做ISR（Interrupt Service Routine）。

#### 7.1 NVIC Controller

Cortex-M中专门用来管理异常的单元模块叫做NVIC（Nested Vectored Interrupt Controller）。在ARM架构中，中断是属于异常的一部分，因为NVIC用来管理异常，故必然也包含着中断。

将外设分为两类，一个在核心之外在MCU之内，另一个完全在MCU之外，而后者的中断是通过I/O口来进行的，而管理这一来源的中断的程序单元为EXTI（External Interrupt/Event Controller），它也将这一从I/O口的信号与NVIC相关联。

宏观的异常可以分为来自于CPU的系统异常和包含上述的两种外设（EXTI和核心之外MCU内的外设）的中断请求**IRQ（Interrupt Requests）**，一般对于中断的操作都是在对NVIC的IRQ上进行操作（只是用NVIC的IRQ这一小部分）。

![image-20220925201123516](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20220925201123516.png)

**中断请求IRQ**
Cortex-M处理器将所有异常分类为十多种的异常，并且对于Cortex-M的产品都是相似的。

其中之一的异常为IRQ（中断请求，所以中断就是异常的一种），并且其又包括了很多种类的中断请求，可以在`\startup_stm32xxx.s`看到

关于异常向量表存放地址没有看懂...

#### 7.2 Enabling Interrupts
STM32 MCU开启时，默认只有Reset，NMI和Hard Fault开启，若要开启属于IRQ的中断请求，可以使用以下函数

```c
void HAL_NVIC_EnableIRQ(IRQn_Type IRQn);
```

IRQn_Type是枚举类型，HAL库已经将所有的中断定义在了枚举类型中。

同样你也可以关闭某个IRQ，使用下面函数
```c
void HAL_NVIC_DisableIRQ(IRQn_Type IRQn);
```

需要注意的是，上述两个函数实在NVIC层面上使能的，由上图可以看到，要实际使用中断模式，还需要用其他的函数联系处理器内核，也就是说，NVIC层面上的打开是使用中断的基础。

比如，打开PA5的中断，会在GPIO的初始化函数`void MX_GPIO_Init(void)`中添加以下代码
```c
void MX_GPIO_Init(void) {
  ...
  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI9_5_IRQn, 0, 0);		// 设置抢占优先级和子优先级
  HAL_NVIC_EnableIRQ(EXTI9_5_IRQn);
}
```

**External Lines and NVIC**

STM32将EXTI所接收到的所有中断请求视为一定数量的中断请求，并以此连接到NVIC中。

EXTI所管理的中断请求是来源于板外的外设的中断请求，所以只能通过GPIO的输入来实现，与此有关的GPIO分为一系列的EXTI lines，并且规定PxN（N是固定数字）表示第N个EXTI line，所以Px0到Px15共有16条EXTI lines，但是只有其中的7条是相互独立的，这里的独立的概念是指这七条是由不同的7个位于NVIC的ISR来管理，如Px0被EXTI0_IRQ所管理，而Px10到Px15由EXTI15_10_IRQ管理（即这些中断的回调函数只有一个），如下图所示
![image-20220927000953107](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20220927000953107.png)

需要注意的是，不能同时将同属一条EXTI line的GPIO口设置为中断来源，**因为ISR只传入了PIN参数而没有传入Port参数**，所以不能分辨出来，换句话说，一条EXTI line只能处理一种中断请求。

**The process of EXTI in HAL**
每当一个EXTI发生后，HAL库首先会调用该EXTI线路对应的ISR（在NVCI定义的IRQHandler函数），如EXTI1对应`void EXTI1_IRQHandler(void);`，EXTI5对应`void EXTI9_5_IRQHandler(void);`。IRQHandler属于Application层面（底层），在该函数的内部，会调用属于ST HAL层面的`void HAL_GPIO_EXTI_IRQHandler(uint16_t GPIO_Pin);`，所有不同的IRQHandler都会调用这个相同的函数，并且传入GPIO_Pin参数（对应着EXTI line）由于一些EXTI line（EXTI9_5_IRQ和EXTI15_10_IRQ）共用的ISR无法区分传入的GPIO_Pin，所以此IRQ函数内部会依次调用对应参数的HAL_IRQHandler函数，并在其内部判断传入的参数是否属于Pending状态，从而判断究竟是哪一个EXTI line引起的中断。找到对应EXTI line后就以GPIO_Pin为参数调用中断回调函数`__weak void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin);`，其中__weak表示允许“重载”（完全代替）。

综上所示，由宏观层面看，无论是哪一个EXTI line发出了中断请求，最终都会调用相同的`HAL_GPIO_EXTI_IRQHandler()`函数，进而调用`HAL_GPIO_EXTI_Callback()`，而区别在于不同的ISR可以区分不同的GPIO_Pin参数。

```c
// stm32f4xx_it.c
void EXTI0_IRQHandler(void)
{
  HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);
}

// stm32f4xx_it.c
void EXTI15_10_IRQHandler(void)
{
  /* USER CODE BEGIN EXTI15_10_IRQn 0 */

  /* USER CODE END EXTI15_10_IRQn 0 */
  HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_10);	// 同属同一EXTI15_10的不同Pin依次检查。
  HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_11);
  HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_13);
  /* USER CODE BEGIN EXTI15_10_IRQn 1 */

  /* USER CODE END EXTI15_10_IRQn 1 */
}

// stm32f4xx_hal_gpio.c
void HAL_GPIO_EXTI_IRQHandler(uint16_t GPIO_Pin)
{
  /* EXTI line interrupt detected */
  if(__HAL_GPIO_EXTI_GET_IT(GPIO_Pin) != RESET)		// 判断是否这个GPIO_Pin有引起中断
  {
    __HAL_GPIO_EXTI_CLEAR_IT(GPIO_Pin);
    HAL_GPIO_EXTI_Callback(GPIO_Pin);
  }
}
```

其实每一个外设产生的中断都要经过两个IRQHandler函数和一个Callback函数，其中只有第一个IRQHandler不属于HAL层。

#### 7.3 Interrupt Lifecycle

从宏观来讲，当一个中断来袭时，它首先会被标记为等待响应（pending），此刻状态为inactive，如果当前没有其他的ISR正在执行，则该中断会清理标记位后执行，此刻状态为active。如果当前有其他的ISR执行，并且新的中断响应的优先级还更低，其就会一直保持pending状态直到优先级高的中断的ISR执行完毕。如果正在执行的ISR优先级更低，则正在执行的ISR会变为inactive（此时没有pending标记了），转而执行新的优先级高的袭来的ISR，直到其完成后返回原来的ISR。如果拥有相同的优先级，则与新中断为低优先级的情况一致，保持pending标记直到完成。

pending标记可以在等待的途中被取消，被取消后就不会执行对应的ISR了。

从更底层的方式来看，每个外设（包括EXTI）都拥有一个与NVIC相连的特殊位，当中断发生时，该位置高（外设的pending标记）以提醒NVIC，在下一个周期，NVIC的pending标记会置高，直到对应中断的ISR执行而**自动**（这意味着不需要实际的代码）清除NVIC的pending，而EXTI的pending会在后面使用对应的代码先进行判断`if(__HAL_GPIO_EXTI_GET_IT(GPIO_Pin) != RESET)`，该函数查询外设的pending标记，如果确实是由该GPIO_Pin引发的，那么就会执行`__HAL_GPIO_EXTI_CLEAR_IT(GPIO_Pin);`来清除外设的pending。在外设的pending没有清除前，是无法继续收到新的相同的中断请求的。

如：GPIOA的Pin1发生中断->EXTI_1保持pending->NVIC保持pending->NVIC执行ISR从而清除NVIC的pending->ISR执行过程中清除EXTI_1的pending。

*在某中断ISR执行的过程中，同一中断来袭会怎么处理呢？*

```c
uint32_t HAL_NVIC_GetPendingIRQ(IRQn_Type IRQn);
void HAL_NVIC_SetPendingIRQ(IRQn_Type IRQn);	
void HAL_NVIC_ClearPendingIRQ(IRQn_Type IRQn);	
uint32_t HAL_NVIC_GetActive(IRQn_Type IRQn);
```

由函数名称可见，上述的函数都是获取与NVIC pending有关的函数。

实际调用`HAL_NVIC_SetPendingIRQ()`函数后应该不会执行callback，因为进入callback之前会判断外设是否处于pending标记，但会确实引发执行对于的ISR。

#### 7.4 Interrupt Priority Levels

中断优先级将会定义以下两件事的执行：

- 两个中断同时（同一时钟周期）发生时应该执行哪个中断。
- 当前的中断是否会被新来的中断给抢占。

**Cortex-M3/4/7**
Cortex-M3/4/7核心的优先级将由一个8位的名字叫IPR寄存器来定义，但是STM32实际只使用其中的高4位（Cortex-M0/0+只用两位，并且只有抢占优先级），这4位可以表示16个等级，数字大小越低，优先级越高。

对于STM32实际使用的4位，还会分为抢占优先级（preemption priority）和子优先级（sub-priority）。抢占优先级顾名思义，当中断同时发生时，抢占优先级高的先执行，当不同时发生时，抢占优先级会将当前的更低抢占优先级暂停。而子优先级是当同时有多个其他的抢占优先级相同的中断同时为pending标记的情况下，其决定着当前ISR执行完后下一个该执行哪个中断ISR。

抢占优先级和子优先级的分法由SCB->AIRCR寄存器来定义，有如下的四种情况
![image-20220929003236501](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20220929003236501.png)

MX代码在`HAL_Init();`中定义了分组
```c
HAL_StatusTypeDef HAL_Init(void)
{
  ...
  /* Set Interrupt Group Priority */
  HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_4);	// 默认是4bit的抢占优先级，即0-15的抢占级别
  ...
}
```

Cortex-M3/4/7的核心可以在打开中断使能后在此设置优先级或者优先级分组（Cortex-M0/0+使能中断后就不能再改变了），但是改变分组的同时，寄存器上的值是不会改变的，所以改变分组后优先级是怎么样的是难以预料的，所以尽量不要随意修改优先级分组，或是对优先级分组后在此重新设置优先级。

HAL中还有一些其他的函数可以获取优先级的有关信息
```c
void HAL_NVIC_GetPriority(IRQn_Type IRQn, uint32_t PriorityGroup, uint32_t *pPreemptPriority, uint32_t *pSubPriority);
```

关于这个函数仅仅只获得优先级为什么还有有分组有关的参数，但是几乎有同样过程的`HAL_NVIC_SetPriority()`函数却不用分组信息，好吧作者也不知道...
细看`HAL_NVIC_SetPriority()`这个函数后，发现其代码内部调用了app层面的`NVIC_GetPriorityGrouping()`来获取分组的消息

```c
uint32_t HAL_NVIC_GetPriorityGrouping(void)
{
  /* Get the PRIGROUP[10:8] field value */
  return NVIC_GetPriorityGrouping();
}
```

这个函数就直接把源代码放出来吧，反正也不长。

*7.5 Interrupt Re-Entrancy*

*这小节没看懂...*

*7.6 Mask All Interrupts at Once or an a Priority Basis*

*这是关于屏蔽所有中断和异常的操作，等到需要用的时候在来看，这里留个心眼儿...*

### 8. Universal Asynchronous Serial Communication

 USART（Universal Synchronous/Asynchronous Receiver/Transmitter），本章只介绍其polling和interrupt模式。

#### 8.1 Introduction to UARTs and USARTs

当我们需要在两个设备传输数据的时候，我们有两种方式进行传输，一是并行的方式，两个设备相连一定数量的线（通常是一个字节的数量8条），同一时间传入一个字节；二是使用串行的方式，将字节中的位连续的传输在一条线中，而UART/USART就是将很多个一连串的位（通常为**1byte**）翻译成连续的属于同一根线的位流。

两个设备传输数据的前提是在时间（timing）上达成了某种一致，在同步传输（synchronous transmission）中，两个设备拥有相同的CLK（而提供这相同的CLK的是两设备之一，通常也称作主机master），并且由主机master开始提供CLK，这也通常意味着为数据传输的开始。我们将这种传输方式称为USART，其至少需要TX、RX和CLK三个接口。

异步方式传输时，两个设备对于传输所需要的时间达成了一致（主机从机设置相同的波特率），所以不需要主机发送CLK，主机的TX端口位于**空闲状态时，输出保持高位**，当要开始传输时，下降成低位并保持1.5个周期（也可能一周期或者两周期），表示这即将开始输送数据，输送完成之后，主机输出保持高位1.5个周期，表示传输完成，后面继续保持高天平的空闲状态。该方式称为UART，只需要TX和RX两个接口（和地线保持参考电压相同）。

上述的UART/USART只提到了传输数据的方式，并没有说明电平的关系，所以当实际使用时，还需要其他的标准，如RS232和RS485。以RS282为例，该标准提供了专用的Hardware Flow Control新路，其多添加了两个接口，Request to Send（RTS）和Clear to Send（CTS），RTS用于发送端告诉接收端开始准备接收数据，而CTS用于由接收端告诉发送端已经准备好接收数据。

#### 8.2 UART Initialization

与GPIO的配置类似，UART同样也映射到内存中的某个区域，HAL库使用USART_TypeDef指针来表示这一区域（就像使用GPIO_TypeDef来表示GPIOx），这里之所以取名USART，是为了表明其即可以接受USART也可以接受UART的外设；同样也有UART_InitTypeDef来初始化UART的配置。与GPIO不同的是，UART将前面的两个结构再次封装成一个结构UART_HandleTypeDef，定义如下
```c
typedef struct __UART_HandleTypeDef
{
  USART_TypeDef                 *Instance;        /*!< UART registers base address        */
  UART_InitTypeDef              Init;             /*!< UART communication parameters      */

  const uint8_t                 *pTxBuffPtr;      /*!< Pointer to UART Tx transfer Buffer */
  uint16_t                      TxXferSize;       /*!< UART Tx Transfer size              */
  __IO uint16_t                 TxXferCount;      /*!< UART Tx Transfer Counter           */
  uint8_t                       *pRxBuffPtr;      /*!< Pointer to UART Rx transfer Buffer */
  uint16_t                      RxXferSize;       /*!< UART Rx Transfer size              */
  __IO uint16_t                 RxXferCount;      /*!< UART Rx Transfer Counter           */
  __IO HAL_UART_RxTypeTypeDef ReceptionType;      /*!< Type of ongoing reception          */
  DMA_HandleTypeDef             *hdmatx;          /*!< UART Tx DMA Handle parameters      */
  DMA_HandleTypeDef             *hdmarx;          /*!< UART Rx DMA Handle parameters      */
  HAL_LockTypeDef               Lock;             /*!< Locking object                     */
  __IO HAL_UART_StateTypeDef    gState;           
  __IO HAL_UART_StateTypeDef    RxState;          
  __IO uint32_t                 ErrorCode;        /*!< UART Error code                    */

} UART_HandleTypeDef;
```

- Instance：表示指向USART_TypeDef的指针，其表示的地址就是某个UART的内存地址。

- Init：用于初始化UART的配置。
- Lock：用来防止同时对于UART的请求，但是作者说Lock这个机制被很多人所诟病。

以下为UART_InitTypeDef的介绍：

```c
typedef struct
{
  uint32_t BaudRate;                
  uint32_t WordLength;                
  uint32_t StopBits;                 
  uint32_t Parity;                    
  uint32_t Mode;                      
  uint32_t HwFlowCtl;                 
  uint32_t OverSampling;              
} UART_InitTypeDef;
```

- BaudRate：波特率，也即单位时间输送多少位，理论上该参数是可以任意设置的，但是对于一定的外设时钟，常常不能以没有误差的方式来产生对应的波特率，所以一般只使用特定的波特率值，你可以从参考手册查到最合适的波特率。
- StopBits：停止位的发送个数1或2，分别对应`UART_STOPBITS_1`和`UART_STOPBITS_2`。
- Parity：奇偶校验位，放在位的MSB端（MSB具体什么以后再查，反正是8位或9位的最后一个位置），当所有位的1的数量位奇数个，则模式位奇校验时位1，偶校验时位0，其他模式大差不差。
- Mode：将UART设置成TX、RX或TX和RX同时。
- HwFlowCtl：增加Hardware Flow Control，即多了RTS、CTS或RTS和CTS同时。
- OverSampling：接受1个位的时间内不光只采样一次，而采样多次（8次或16次），以减小噪声的干扰。若UART的时钟频率从位48MHz，选择`UART_OVERSAMPLING_16`时最高的波特率为48M/16=3000000bps，选择`UART_OVERSAMPLING_8`时为48M/8=6000000bps。

**The process of UART initialization in HAL**
MX将不同的UART/USART当作不同的外设单独设定初始化函数来调用。对于某个单独的模块（以USART2为例），调用初始化函数`MX_USART2_UART_Init()`，在其内部首先已经有了一个全局定义的`UART_HandleTypeDef`类型（huart2），这个handle包含了对于外设的地址和初始化配置等等其他参数，所以对其类型的成员分别进行赋值（像`huart4.Instance = USART4`等等），然后再调用`HAL_UART_Init()`函数来初始化（对应`HAL_UART_Init(&huart2)`）。

需要注意的是，UART是通过GPIO的复用功能来与外界联系的，所以还需要配置对于的GPIO端口，MX在其`HAL_UART_Init()`函数中自动调用了`HAL_UART_MspInit()`函数，该函数中包括了对对应GPIO的配置和DMA配置（如果打开了DMA的话）。

综上，初始化的过程为`MX_USART2_UART_Init() ->  HAL_UART_Init() -> HAL_UART_MspInit()` ，其中`HAL_UART_MspInit()`函数在`usart.c`中进行了重载。

```c
// usart.c
void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;		// 定向USART2的地址
  // 初始化参数设置
  huart2.Init.BaudRate = 115200;	
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)		// 使用初始化参数进行初始化
  {
    Error_Handler();
  }
}
```

#### 8.3 UART Communication in Polling Mode

polling mode，也称阻塞模式（blocking mode），使用该模式时，系统会等待其发送或接收数据完成才会进行下一步的程序，所以常常是用与速度较快的传输并且该传输不常用，如调试的时候。

通过以下函数来使用UART的阻断模式
```c
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *huart, const uint8_t *pData, 
                                    uint16_t Size, uint32_t Timeout);
HAL_StatusTypeDef HAL_UART_Receive(UART_HandleTypeDef *huart, uint8_t *pData, 
                                   uint16_t Size, uint32_t Timeout);
```

其参数不再介绍，需要注意的是pData的类型为uint8_t的指针，而Size为uint16_t类型的整数。

`HAL_UART_Transmit()`机理实际上就是不断的为`huart->Instance->DR`按顺序赋值上`pData`数组，置于如何控制波特率，我还不知道。

```c
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *huart, const uint8_t *pData, 
                                    uint16_t Size, uint32_t Timeout)
{
    // ...
    while (huart->TxXferCount > 0U)
    {
      // ...
      huart->Instance->DR = (uint16_t)(*pdata16bits & 0x01FFU);
      pdata16bits++;
      huart->TxXferCount--;
    }
    // ...
}
```

`HAL_UART_Receive()`完全相反，按顺序循环给pData赋值上`huart->Instance->DR`里量。

```c
HAL_StatusTypeDef HAL_UART_Receive(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size, uint32_t Timeout)
{
	// ...
    /* Check the remain data to be received */
    while (huart->RxXferCount > 0U)
    {
      // ...
      *pdata16bits = (uint16_t)(huart->Instance->DR & 0x01FF);
      pdata16bits++;
      huart->RxXferCount--;
    }
	// ...
  }
}
```



#### 8.4 UART Communication in Interrupt Mode

在使用UART的中断模式之前，你必须先开启USARTx_IRQn，并设置中断优先级。

关于UART的中断，系统提供了其很多个中断源，包括发送和接收，比如发送完成，接收完成等等，对于每种中断源都有对应的中断回调函数，所以你只需要将所需要的代码放入对应的终端回调函数中即可。
```c
HAL_StatusTypeDef HAL_UART_Transmit_IT(UART_HandleTypeDef *huart, const uint8_t *pData, uint16_t Size)
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size);
```

上述函数允许时，会很快的将控制权传入主程序（此时并没有完成发送和接收），当真正的发送和接收完成时，就会立马产生中断从而调用对应的ISR，并调用对应的终端回调函数，一般时发送或接收完成触发的中断源
```c
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart);
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart);
```

举个例子
```c
typedef enum {
	USART_RESET = 0,
	USART_SET = 1
} UART_ReadyTypeDef;
UART_ReadyTypeDef UsartReady = USART_SET;

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
	UsartReady = USART_SET;
}

void printWelcomeMessage(void) {

	const char *WELCOME_MSG = "This is a UART test program\r\n";
	const char *MAIN_MENU = "Select the option you are interested in:\r\n"
							"\t1. Toggle D2 LED\r\n"
							"\t2. Read USER BUTTON status\r\n"
							"\t3. Clear screen and print this message again\r\n";
	const char *string[] = {"\033[0;0H", "\033[2J", WELCOME_MSG, MAIN_MENU};

	for (uint8_t i = 0; i < 4; ++i) {
		HAL_UART_Transmit_IT(&huart2, (uint8_t *)string[i], strlen(string[i]));
		while (HAL_UART_GetState(&huart2) == HAL_UART_STATE_BUSY_TX ||
			   HAL_UART_GetState(&huart2) == HAL_UART_STATE_BUSY_TX_RX) {
			;
		}
	}
}

uint8_t readUserInput(void) {
	const char *PROMPT = "please select your choice: ";
	uint8_t retVal = 0;
	static char readBuf[2] = {0};

	HAL_UART_Transmit(&huart2, (uint8_t *)PROMPT, strlen(PROMPT), HAL_MAX_DELAY);

	if (UsartReady == USART_SET) {
		UsartReady = USART_RESET;
		HAL_UART_Receive_IT(&huart2, (uint8_t *)readBuf, 2);

		retVal = (readBuf[1] = '\0', atoi(readBuf));
	}
	return retVal;
}

int main() {
    ...
printMessage:
	printWelcomeMessage();
	while (1) {
		opt = readUserInput();
		if (opt > 0) {
			if (opt == 3) {
				goto printMessage;
			}
		}
		otherThings();
	} 
    ...
}
```

如果只是用阻塞模式，那么你会不断的等待直到用户输入，此时此刻你不能干别的事情，但是如果你用中断模式，你不需要等待，而是可以转而去干其他事情。

比如上述例子，定义了一个全局变量UsartReady，当其为SET时，代表用户已经发送，当其为RESET时，表示等待用户的发送，初始化为SET是为了先执行依次中断模式的接收函数以开启真正的中断。

在每次进行otherThing之前，先判断是否UsartReady为SET，若为SET，则表示输入完成，可以进入处理用户的输入，若为RESET，则用户没有输入完成，此时进行otherThing。

注意，上述代码对于输入的格式有着严格的要求，用户稍微输入错误的格式就会引发程序错误，解决的方法是处理多余的输入，但是我还不会...

这一章后面没有学仔细，以后再看吧...

**UART Related Interrupts**
与UART有关的中断源有很多，但是STM32只为其提供一个ISR（就像多条EXTI Line却只对应一个ISR），但是HAL库中的ISR中帮我们写好了判断的代码（在`HAL_UART_IRQHandler()`函数中），这个ISR中自动判断并调用对应的中断回调函数，我们只需调用对应中断源的中断回调函数即可。



### 9. DMA Management

DMA（Direct Memory Access）为每个外设都开放了一个直接通向内存的通道，从而免去了CPU管理数据传输时的开销（当然有些比笔要的开销是用于DMA的配置）。

#### 9.1 Introduction to DMA

有三种DMA框架，但是ST官方又将这三种框架赋予了不同的术语，导致我们以为三种框架是完全不同的东西，但实际上，不同的框架之间的差异仅仅在于某些功能自由度不同。

**The Need of a DMA and the Role of the Internal Buses**
理论上MCU可以设计为其内的每个外设都有一个专属的存储区域的样式，由此每当CPU要使用数据时，都直接访问那个外设的专属区域即可，但是增加了许许多多的的复杂性和功耗。所以MCU使用其内部的SRAM中专门的区域来暂时存放不同外设产生的数据，如

```c
uint8_t buf[20];
HAL_UART_Receive(&huart2, buf, 20, HAL_DELAY);
```

我们通过buf数据来开创了一个在SRAM中的空间，由此来暂时存放由UART传输来的数据，并且CPU会一直在这里被占用，直到传输完成，这非常的占用CPU的资源，所以才需要DMA，从而外设拥有了与SRAM的直接数据传输通道。

让我们再次回顾这张图

![image-20220924111718434](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20220924111718434.png)

首先CPU和DMA作为真个系统的主机轮番控制着bus matrix的控制权，并通过一系列的线路来与各个部件相连。
其次BusMatrix管理着CPU和DMA的分配，并且连接着CPU、DMA和其他四个组件，其允许各个组件的相互交流。
由于从机没有管理bus的能力，所以DMA requests是用来让从机提醒DMA从而开启DMA的。

#### 9.2 `HAL_DMA` Module

HAL库的所有外设都是使用一个句柄来表示，其中DMA就是用`DMA_HandleTypeDef`类型来表示的。

**`DMA_HandleTypeDef` in F0/F1/F3/L0/L4 HALs**
在标题中的系列中，其`DMA_HandleTypeDef`定义如下

```c
typedef struct {
    DMA_Channel_TypeDef *Instance; /* Register base address */
    DMA_InitTypeDef Init; /* DMA communication parameters */
    HAL_LockTypeDef Lock; /* DMA locking object */
    __IO HAL_DMA_StateTypeDef State; /* DMA transfer state */
    void *Parent; /* Parent object state */
    void (* XferCpltCallback)( struct __DMA_HandleTypeDef * hdma);
    void (* XferHalfCpltCallback)( struct __DMA_HandleTypeDef * hdma);
    void (* XferErrorCallback)( struct __DMA_HandleTypeDef * hdma);
    __IO uint32_t ErrorCode; /* DMA Error code */
} DMA_HandleTypeDef;
```

- Instance：是一个指向`DMA_Channel_TypeDef`变量的指针，其代表着某个DMA的某一个通道（把单独的通道当为独立的个体），记住，每个DMA的每个通道都有对应的外设与其绑定，意味着某个外设的DMA请求必须由于其相对应的DMA通道来处理；
- Init：是一个`DMA_InitTypeDef`类型的变量，用于初始化该DMA的该通道的配置；
- Parent：用于追踪该DMA所服务的外设，如开启了UART的DMA模式，则此时Parent为指向`UART_HandleTypeDef`类型的指针；
- XferxxxCallback：用于完成DMA传输后的回调函数。

现在介绍DMA的初始化配置，由类型`DMA_InitTypeDef`来完成
```c
typedef struct {
    uint32_t Direction;
    uint32_t PeriphInc;
    uint32_t MemInc;
    uint32_t PeriphDataAlignment;
    uint32_t MemDataAlignment;
    uint32_t Mode;
    uint32_t Priority;
} DMA_InitTypeDef;
```

- Direction：用来表示DMA传输的方向，即外设到内存、内存到外设和内存到内存；
- PeriphInc：外设存储数据的寄存器是否每次传输都要指针自增，一般不自增，因为外设数据寄存器就这么小；
- MenInc：存储在内存的数据在DMA传输时是否要自增，一般都要自增，不然就都收到同一个地址上了；
- PeriphDataAlignment, MemDataAlignment：即一次大传输（整个数组）的一次小传输（数组元素的大小）的字节大小，1byte、1word（4byte）或half word（2byte）；
- Mode：选择正常模式还是循环模式，有些不懂；
- Priority：优先级。

**DMA_HandleTypeDef in F2/F4/F7 HALs**
在上述标题系列的单片机中，只有`DMA_InitTypeDef`有所增加

```c
typedef struct {
    uint32_t Channel;
    uint32_t Direction;
    uint32_t PeriphInc;
    uint32_t MemInc;
    uint32_t PeriphDataAlignment;
    uint32_t MemDataAlignment;
    uint32_t Mode;
    uint32_t Priority;
    uint32_t FIFOMode;
    uint32_t FIFOThreshold;
    uint32_t MemBurst;
    uint32_t PeriphBurst;
} DMA_InitTypeDef;
```

即将DMA和其的通道区分开了，所以此时一个`DMA_HandleTypeDef`只代表具体的DMA而不代表具体的通道，具体的通道在`DMA_InitTypeDef`中的`Channel`来表示。

- FIFOMode：FIFO就是在DMA传输的目的地之前增加一个缓冲，由此而带来一系列的好处，没很看懂；
- FIFOThreadhold：即FIFO的缓冲为多少时，将FIFO内的数据发送目的地；
- MemBurst：没看懂；
- PeriphBurst：没看懂。

**DMA Transfers in Polling Mode**
要使用DMA传输数据，在完成初始化设置之后，我们首先需要做以下几步

- 确定源地址发目的地地址；
- 确定要传输数据的大小；
- 装备DMA；
- 启动DMA。

其中可以使用下面的函数可以完成前三步
```c
HAL_StatusTypeDef HAL_DMA_Start(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t DataLength)
```

第四步启动DMA要从外设层面上开启，一般是直接通过操控对应外设的某个寄存器来进行的，也就是说，HAL库简化后，DMA传输只需要两步

- 使用`HAL_DMA_Start()`函数装备DMA，其中包含源地址、目标地址和数据传输的大小；
- 在对应外设中启动DMA。

以下为具体例子

```c
UART_HandleTypeDef huart2;
DMA_HandleTypeDef hdma_usart2_tx;
char msg[] = "Hello STM32 Lovers! This message is transferred in DMA Mode.\r\n";

int main(void) {
    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* Configure the system clock */
    SystemClock_Config();

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_USART2_UART_Init();
    MX_DMA_Init();

    /* USART2_TX Init */
    /* USART2 DMA Init */	// 只有Init成员被初始化时使用
    hdma_usart2_tx.Instance = DMA1_Channel4;
    hdma_usart2_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_usart2_tx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart2_tx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart2_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart2_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart2_tx.Init.Mode = DMA_NORMAL;
    hdma_usart2_tx.Init.Priority = DMA_PRIORITY_LOW;
    HAL_DMA_Init(&hdma_usart2_tx);
    
	// 装备DMA
    HAL_DMA_Start(&hdma_usart2_tx, (uint32_t)msg, (uint32_t)&huart2.Instance->TDR, strlen(msg));
    //Enable UART in DMA mode
    huart2.Instance->CR3 |= USART_CR3_DMAT;
    //Wait for transfer complete
    HAL_DMA_PollForTransfer(&hdma_usart2_tx, HAL_DMA_FULL_TRANSFER, HAL_MAX_DELAY);
    //Disable UART DMA mode
    huart2.Instance->CR3 &= ~USART_CR3_DMAT;
    //Turn LD2 ON
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);

    /* Infinite loop */
    while (1);
}
```

该段程序为使用DMA将数据传输至GPIO口的数据寄存器从而实现亮灯操作，在该程序中，首先通过`MX_USART2_UART_Init()`进行了UART的基础初始化配置，但此时还没有进行UART的DMA初始化（目前MX配置中的UART初始化函数中已经包含了对应外设的DMA的初始化），然后对DMA进行初始化（注意只是DMA，而不是UART的DMA，其实就是开启DMA的中断），再然后手动通过对`DMA_HandleTypeDef`变量进行初始化配置从而出初始化了UART的DMA，至此，初始化完成（在目前的CubeMX中，上述操作都集成在了两个函数当中，分别是`  MX_DMA_Init()`和`  MX_UART4_Init()`）。

在初始化配置完成之后，就将要按照前述的四步进行DMA传输，首先通过`HAL_DMA_Start()`函数来完成前面三步，即确定源地址和目标地址、确定数据大小和装备DMA（开启DMA的必要条件），最后一步通过在外设的寄存器来开启外设的DMA，从而开启了整个DMA过程。

由于DMA的传输无关核心，开启了DMA后，就直接转到下一行代码了，所以无法知道其传输完成的时间，可以通过使用函数`HAL_DMA_PollForTransfer()`来等待传输的完成，DMA完成之后，该函数会自动关闭外设的DMA（即该外设的DMA标志位设置为空闲）。

但是DMA的阻塞模式实际上是没有意义的，因为使用DMA的原因就是将CPU从传输数据的任务解放出来，阻塞模式虽然交给了DMA来传输数据，但是CPU仍然需要等待完成，所以还得需要中断模式的DMA。

**DMA Transfers in Interrupt Mode**
在对应DMA和外设初始化完成之后，开启DMA的中断模式需要以下步骤

- 设置回调函数（即让DMA句柄中的回调函数指针指向某个自己定义的函数）；
- 编写回调函数的代码；
- 打开NVIC中的DMA的中断使能；
- 用`HAL_DMA_Start_IT()`函数开启DMA的中断模式（和阻塞模式中的前三步骤一样）；
- 配置外设的寄存器开启DMA传输。

以下为DMA中断模式示例

```c
int main(void) {
    hdma_usart2_tx.Instance = DMA1_Channel4;
    hdma_usart2_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_usart2_tx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart2_tx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart2_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart2_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart2_tx.Init.Mode = DMA_NORMAL;
    hdma_usart2_tx.Init.Priority = DMA_PRIORITY_LOW;
    // 设置回调函数，该项不属于初始化的内容
    hdma_usart2_tx.XferCpltCallback = &DMATransferComplete;
    HAL_DMA_Init(&hdma_usart2_tx);

    /* DMA interrupt init */
    // 此时只需要打开DMA的中断使能，而不用打开外设的中断使能，因为只是用DMA方面的函数没有将外设的标志位设置为忙碌
    HAL_NVIC_SetPriority(DMA1_Channel4_5_IRQn, 0, 0);	
    HAL_NVIC_EnableIRQ(DMA1_Channel4_5_IRQn);

    HAL_DMA_Start_IT(&hdma_usart2_tx, (uint32_t)msg, (uint32_t)&huart2.Instance->TDR, strlen(msg));

    //Enable UART in DMA mode
    huart2.Instance->CR3 |= USART_CR3_DMAT;	//真正产生DMA请求

    /* Infinite loop */
    while (1);
}

// 自己设置的中断回调函数
void DMATransferComplete(DMA_HandleTypeDef *hdma) {
    if(hdma->Instance == DMA1_Channel4) {
        //Disable UART DMA mode
        huart2.Instance->CR3 &= ~USART_CR3_DMAT;
        //Turn LD2 ON
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    }
}
```

其实相比于阻塞模式，也就多了一个回调函数的配置、开启中断和少了一个等待的函数，多了一个回调函数。

**Using the `HAL_UART` Module with DMA Mode Transfers**
上述操作（从DMA外设的角度触发）实现DMA传输的方式还是略微复杂，在开启DMA传输的时候还要涉及到外设的底层寄存器，不过HAL库为我们在外设层面更加抽象了一层DMA操作，可以直接通过使用外设的DMA传输函数`HAL_UART_Transimit_DMA()`来进行DMA传输（从外设的角度出发，DMA只是外设的一个功能），实际步骤如下

- 在UART的初始化中不光配置UART本身的初始化，还要配置UART对应DMA的初始化，其在`HAL_UART_MspInit()`函数中完成；
- 使用`__HAL_LINKDMA()`来完成DMA与外设的更底层连接；
- 打开DMA中断使能；
- 打开外设中断使能（一定要打开，因为虽然是DMA模式，但是DMA的中断只负责DMA相关标志的清理，而UART相关的清理是在UART中断函数进行的，所以在UART的DMA传输模式每次传输完成后，会触发两次中断回调，一次DMA的中断回调，其清除相关DMA的标志位；另一次UART的中断回调，其清理相关UART的标志位。只有当UART和DMA的标志位都清理了才能开启下一次传输，不这样做的话就可能造成只进行了一次传输的错误）。

**所以不应当将外设的中断模式与外设的DMA模式区分开为独立的模式，而应该视外设的DMA模式是建立在外设的中断模式之上的。**

以下为示例
```c
uint8_t dataArrived = 0;
uint8_t data[3];
int main(void) {
    HAL_Init();
    Nucleo_BSP_Init(); // Configure the UART2
    // Configure the DMA1 Channel 5, which is wired to the UART2_RX request line
    hdma_usart2_rx.Instance = DMA1_Channel5;
    hdma_usart2_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_usart2_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart2_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart2_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart2_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart2_rx.Init.Mode = DMA_NORMAL;
    hdma_usart2_rx.Init.Priority = DMA_PRIORITY_LOW;
    HAL_DMA_Init(&hdma_usart2_rx);
    // Link the DMA descriptor to the UART2 one
    __HAL_LINKDMA(&huart, hdmarx, hdma_usart2_rx);	// 底层将DMA与外设连接，hdmarx是结构体huart的一个成员
    /* DMA interrupt init */
    HAL_NVIC_SetPriority(DMA1_Channel4_5_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Channel4_5_IRQn);
    /* Peripheral interrupt init */
    HAL_NVIC_SetPriority(USART2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
    // Receive three bytes from UART2 in DMA mode
    HAL_UART_Receive_DMA(&huart2, &data, 3);
    while(!dataArrived); // Wait for the arrival of data from UART
    /* Infinite loop */
    while (1);
}
// This callback is automatically called by the HAL when the DMA transfer is completed
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
	dataArrived = 1;	
}
void DMA1_Channel4_5_IRQHandler(void) {
    HAL_DMA_IRQHandler(&hdma_usart2_rx); // This will automatically call the HAL_UART_RxCpltCallback()
}
```

实际上直接使用`HAL_UART_Transmit_DMA()`函数所做之事与上述使用DMA的中断模式无差，都是先配置UART和DMA的初始化，然后添加回调函数。

外设的中断究竟和DMA的中断如何纠缠在一起的呢？
对于外设的中断（以UART为例），UART在发出中断请求后，最先调用的是`UART4_IRQHandler()`函数，该函数再调用`HAL_UART_IRQHandler(&huart4)`函数，而在该函数中就包含了大量的判断，和将UART设置为空闲状态，以便开启下一次传输，最后再调用中断回调函数。

对于外设的DMA中断，完成传输后，DMA会发送中断请求，则最先开始调用`DMA1_Stream4_IRQHandler()`函数，其调用`HAL_DMA_IRQHandler(&hdma_uart4_tx)`函数，其中也包含了大量的判断，并且也会将DMA设置为空闲状态，以便开启下一次DMA，最后在调用中断回调函数，只不过该中断回调函数与外设中断最终调用的中断回调函数完全一样。

只有两个中断回调都调用，才能将UART和DMA都设置为空闲状态，才能开启下一次的传输。


### 10. Clock Tree

在同一片单片机内，为了达到节省功耗等目的，不同的外设和组件需要不同的时钟，而时钟树就是用来管理时钟的分层组件。

#### 10.1 Clock Distribution

STM32的时钟源可分为高速时钟源和低速时钟源，高速时钟源分为高速外部时钟源（High Speed External，HSE）和RC振荡产生的高速内部时钟源（High Speed Internal，HSI），其中外部时钟源往往拥有更高的频率稳定度，并且一些高速外设需要专门的确定的外部时钟源；低速时钟源同样也分为低速外部时钟源（Low Speed external，LSE）和由RC振荡产生的低速内部时钟源（Low Speed internal，LSI），并且LSI用来专门驱动RTC和IWDT外设。

时钟源通过锁相环（Phase-Locked Loops，PLL）和分频器来组成一个复杂的时钟网，称之为时钟树来满足核心和外设不同的时钟需求。

**Overview of the STM32 Clock Tree**
时钟树的管理是通过RCC（Reset and Clock Reset）外设来配置和控制的，其的配置分为三步

- 选择想要的HSI（内部RC振荡）和HSE（外部晶振），若选择HSE，则需要正确的配置它；
- 若要增加核心的时钟频率，即SYSCLK，则可以配置锁相环来得到比高速时钟（HSI或HSE）频率更高的信号PLLCKL；
- 配置SW（System Clock Switch）和分频器来为不同的组件选择不同的时钟源。

想要自己来配置正确的时钟树是一件十分困难的事情，但是我们可以通过CubeMX来完成。

**MSI**
MSI（Multi Speed Internal）是用来在低功耗运行时的时钟源，其可以提供多种不同的时钟频率，并且配置时间也很快。
![image-20230410163008934](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230410163008934.png)

上图就是选择不同的时钟源所需要消耗的功率、频率精确度和配置时间，可以看到，MSI时功耗最低和配置时间最短的。

**Configuring Clock Tree Using CubeMX**
在默认的MX工程中，MX会自动选择HSI作为SYSCLK，而不通过HSE或HSI的PLL来作为SYSCLK。

系统时钟SYSCLK是由HSI、HSE和PLLCLK（PLLCLK又是由HSI和HSE决定的）三选一决定的，HCLK是用来给高速总线的时钟，也是在MX配置最高时钟的地方，直接在HCLK这一栏上写上最大的时钟频率，按下回车即可自动配置正确的时钟树。

若要使用HSE，则需要现在RCC外设中开启，在HSE中这一栏中，有三个选项可以选择

- Disable：不使用HSE，而使用HSI，此时在Clock Tree中无法选择HSE；
- Crystal/Ceramic Resonator：使用外部的晶体振荡器或陶瓷振荡器，此时系统会自动打开RCC_OSC_IN和RCC_OSC_OUT引脚（若上述选项为LSE，则会开启RCC_OSC32_IN和RCC_OSC32_OUT引脚）；
- BYPASS Clock Source：使用外部的有源振荡器来作为时钟源，此时内部的振荡器仿佛被旁路了，所以叫做旁路模式，这个模式用得不多，了解不深。

RCC同样还有MCO（Mater Clock Output）模式，此时会打开一个引脚，来为其他的有源器件来提供时钟。

#### 10.2 Overview of the `HAL_RCC` Module

配置RCC主要通过两个结构体`RCC_OscInitTypeDef`和`RCC_ClkInitTypeDef`，前者用于SYSCLK之前的时钟配置，包括选择HSI、HSE和PLL和PLL由HSI还是HSE提供等配置，由于整个时钟树中只有前面的部分有PLL，所以结构体变量`RCC_PLLInitTypeDef`也包含在该结构体其中；后者用于SYSCLK之后的配置，包括AHB等各种APB外设的时钟。

在主函数中，MX通过`SystemClock_Config()`函数来进行时钟树的配置。
```C
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}
```

**Compute the Clock Frequency at Run-Time**
没有了参考外部频率，STM32是无法精确的知道自己的时钟频率的，但是可以通过`HAL_RCC_GetSysClockFreq()`函数来获取时钟树的配置，从而得到时钟频率SYSCLK，比如选择了时钟树选择了HSI，那该函数直接返回内部RC振荡器的频率，即宏`HSI_VALUE`，又或者选择了HSE，则返回`HSE_VALUE`宏。

注意，若自己重新配置了时钟树配置，一定要再次调用CMSIS中的`SystemCoreClockUpdate()`函数（这个函数由`HAL_RCC_ClockConfig`函数自动调用了），依次来更新时钟树的参数。

还有注意，实际的Cortex-M频率实际是HCLK，即SYSCLK经过AHP分频后的时钟，所以用`HAL_RCC_GetSysClockFreq()`/AHB-prescaler才是实际的核心频率。

**Enabling the Master Clock Output**
STM32可以作为外部的时钟源来供其他的外设使用，即使用MCO（Master Clock Output）功能，通过调用下列函数

```c
void HAL_RCC_MCOConfig(uint32_t RCC_MCOx, uint32_t RCC_MCOSource, uint32_t RCC_MCODiv);
```

更加详细的介绍去看data sheet。

**Enabling the Clock Security System**
CSS（Clock Security System）是用来检测RCC故障的机制，若发现HSE故障，系统则会自动转化为HSI时钟，并且发生NMI异常，并调用对于的ISR。可以通过`HAL_RCC_EnableCSS()`函数来开启CSS。并且有以下ISR

```c
void NMI_Handler(void) {
	HAL_RCC_NMI_IRQHandler();
}

void HAL_RCC_CSSCallback(void) {
	//Catch the HSE failure and take proper actions
}
```

**HSI Calibration**
HSI Calibration即内部高速时钟的校准，由于内部的RC振荡器频率的精度只能达到百分之一，所以仍需要进行校准，其内部拥有专门用于校准HSI的寄存器，根据具体的内部RC振荡来调整其的值。
