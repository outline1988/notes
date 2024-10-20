### 15. SPI

#### 15.1 Introduction to the SPI Specification

SPI（Serial Peripheral Interface）用于主机与从机之间的同步通信，一个典型的SPI总线包含以下四个信号

- SCK：主机生成时钟供主机和从机通信，该时钟信号必须由主机产生，所以也意味着SPI必须由主机开始通信。SCK的时钟通常是MHz级别，所以SPI通信很快；
- MOSI：即Master Output Slave input，主机发送数据，从机接收数据，SPI有共有两条数据线用于主机与从机的通信；
- MISO：即Master Input Slave Output，从机发送数据，主机接收数据，不同的传输方向使用不同的两根线，由此可以实现全双工通讯（同一时间，允许两个方向进行通信）；
- SSn：即Slave Select，其中n代表着在同一通信中可以有多个从机与同一主机一起参与。在同一时间内，只能使能一个片选信号，也就是说，在同一时间内，数据总线只存在主机与其中一个从机的通信，由此可以使多个从机共享一个数据总线，如果主机和从机各只有一个，SS可以忽略（保持接地）。

当时钟开始振荡，SPI的传输就即刻开始，同时将目标从机的片选SS值低。主机和从机各有一个1word的寄存器（通常是8bit，也有16bit的从机），MOSI从数据最高位开始传输（MSB），而MISO从数据最低位开始传输（LSB），当数据传输完成，时钟停止。若有更多的数据需要传输，则重复上述过程。

**Clock Polarity and Phase**
时钟的极性（Polarity）和相位（Phase）决定着时钟的形状和触发的状态，其也被称为CPOL和CPHA。

CPOL=0时，时钟的默认值为0，CPOL=1时，时钟默认值为1。
CPHA=0时，上升沿触发，CPHA=1时，下降沿触发。

可以总结为下图
![image-20230402225826341](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230402225826341.png)

**Slave Select Signal Management**
主机开始向某一从机传输数据的前提条件（必要不充分条件）是该从机的片选信号置低，STM32提供两种不同的模式处理SS信号，这两种不同的模式统称为NSS：

- NSS software mode：STM32内部的SPI外设不专门提供SS信号，而需要使用其他的信号（如其他的GPIO口等）来作为SS信号，这样做的好处是可以灵活的使用SS口来选择是否需要传输信号。
- NSS hardware mode：STM32内部的SPI外设提供专门的SS信号，其又可进行两项配置
  - NSS output enabled：只能用于自己为主机且只有一个从机的情况，主机开始通信即自动置SS为低，知道关闭SPI才拉高（我也不知道是传输结束还是关闭整个SPI，盲猜传输结束）。
  - NSS output disabled：也就是NSS input disabled，也就是自己作为从机的SS信号，其他主机通过控制该SS信号选择是否向其进行传输。

**SPI TI Mode**
德州仪器公司所创建的SPI协议，没看懂。

#### 15.2 `HAL_SPI` Module

 与STM32的其他外设一样，每一个spi外设都由一个`SPI_HandleTypeDef`变量来控制，其定义如下

```c
/**
  * @brief  SPI handle Structure definition
  */
typedef struct __SPI_HandleTypeDef
{
  SPI_TypeDef                *Instance;      /*!< SPI registers base address               */

  SPI_InitTypeDef            Init;           /*!< SPI communication parameters             */

  uint8_t                    *pTxBuffPtr;    /*!< Pointer to SPI Tx transfer Buffer        */

  uint16_t                   TxXferSize;     /*!< SPI Tx Transfer size                     */

  __IO uint16_t              TxXferCount;    /*!< SPI Tx Transfer Counter                  */

  uint8_t                    *pRxBuffPtr;    /*!< Pointer to SPI Rx transfer Buffer        */

  uint16_t                   RxXferSize;     /*!< SPI Rx Transfer size                     */

  __IO uint16_t              RxXferCount;    /*!< SPI Rx Transfer Counter                  */

  void (*RxISR)(struct __SPI_HandleTypeDef *hspi);   /*!< function pointer on Rx ISR       */

  void (*TxISR)(struct __SPI_HandleTypeDef *hspi);   /*!< function pointer on Tx ISR       */

  DMA_HandleTypeDef          *hdmatx;        /*!< SPI Tx DMA Handle parameters             */

  DMA_HandleTypeDef          *hdmarx;        /*!< SPI Rx DMA Handle parameters             */

  HAL_LockTypeDef            Lock;           /*!< Locking object                           */

  __IO HAL_SPI_StateTypeDef  State;          /*!< SPI communication state                  */

  __IO uint32_t              ErrorCode;      /*!< SPI Error code                           */

#if (USE_HAL_SPI_REGISTER_CALLBACKS == 1U)
  void (* TxCpltCallback)(struct __SPI_HandleTypeDef *hspi);             /*!< SPI Tx Completed callback          */
  void (* RxCpltCallback)(struct __SPI_HandleTypeDef *hspi);             /*!< SPI Rx Completed callback          */
  void (* TxRxCpltCallback)(struct __SPI_HandleTypeDef *hspi);           /*!< SPI TxRx Completed callback        */
  void (* TxHalfCpltCallback)(struct __SPI_HandleTypeDef *hspi);         /*!< SPI Tx Half Completed callback     */
  void (* RxHalfCpltCallback)(struct __SPI_HandleTypeDef *hspi);         /*!< SPI Rx Half Completed callback     */
  void (* TxRxHalfCpltCallback)(struct __SPI_HandleTypeDef *hspi);       /*!< SPI TxRx Half Completed callback   */
  void (* ErrorCallback)(struct __SPI_HandleTypeDef *hspi);              /*!< SPI Error callback                 */
  void (* AbortCpltCallback)(struct __SPI_HandleTypeDef *hspi);          /*!< SPI Abort callback                 */
  void (* MspInitCallback)(struct __SPI_HandleTypeDef *hspi);            /*!< SPI Msp Init callback              */
  void (* MspDeInitCallback)(struct __SPI_HandleTypeDef *hspi);          /*!< SPI Msp DeInit callback            */

#endif  /* USE_HAL_SPI_REGISTER_CALLBACKS */
} SPI_HandleTypeDef;
```

- Instance：用于指向spi的具体地址；
- Init：专门用来初始化spi的变量；

其中`SPI_InitTypeDef`变量专门用来初始化具体的spi外设，其定义如下

```c
/**
  * @brief  SPI Configuration Structure definition
  */
typedef struct
{
  uint32_t Mode;                /*!< Specifies the SPI operating mode.
                                     This parameter can be a value of @ref SPI_Mode */

  uint32_t Direction;           /*!< Specifies the SPI bidirectional mode state.
                                     This parameter can be a value of @ref SPI_Direction */

  uint32_t DataSize;            /*!< Specifies the SPI data size.
                                     This parameter can be a value of @ref SPI_Data_Size */

  uint32_t CLKPolarity;         /*!< Specifies the serial clock steady state.
                                     This parameter can be a value of @ref SPI_Clock_Polarity */

  uint32_t CLKPhase;            /*!< Specifies the clock active edge for the bit capture.
                                     This parameter can be a value of @ref SPI_Clock_Phase */

  uint32_t NSS;                 /*!< Specifies whether the NSS signal is managed by
                                     hardware (NSS pin) or by software using the SSI bit.
                                     This parameter can be a value of @ref SPI_Slave_Select_management */

  uint32_t BaudRatePrescaler;   /*!< Specifies the Baud Rate prescaler value which will be
                                     used to configure the transmit and receive SCK clock.
                                     This parameter can be a value of @ref SPI_BaudRate_Prescaler
                                     @note The communication clock is derived from the master
                                     clock. The slave clock does not need to be set. */

  uint32_t FirstBit;            /*!< Specifies whether data transfers start from MSB or LSB bit.
                                     This parameter can be a value of @ref SPI_MSB_LSB_transmission */

  uint32_t TIMode;              /*!< Specifies if the TI mode is enabled or not.
                                     This parameter can be a value of @ref SPI_TI_mode */

  uint32_t CRCCalculation;      /*!< Specifies if the CRC calculation is enabled or not.
                                     This parameter can be a value of @ref SPI_CRC_Calculation */

  uint32_t CRCPolynomial;       /*!< Specifies the polynomial used for the CRC calculation.
                                     This parameter must be an odd number between Min_Data = 1 and Max_Data = 65535 */
} SPI_InitTypeDef;
```

其每项的成员就是MX中具体需要配置的参数，介绍如下

- Mode：用于选择主机还是从机；
- Direction：用于选择全双工还是半双工（4-wires），和只接收和只发送（3-wires）