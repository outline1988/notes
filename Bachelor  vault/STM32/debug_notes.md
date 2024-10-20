## 关于C/C++的编译

### make/makefile/cmake到底是什么

对于一个不小型的C/C++项目，其内包含着大量的源文件和头文件，并且有可能还需要以来很多的第三方库。我们通常使用编译器来将这个工程进行编译，如GNU Compiler Collection（编译器的话题等会儿再聊），若工程中只要少量的源文件时，可以直接使用gcc命令来对项目进行编译，但是如果该项目有很多的源文件，甚至还包含着第三方依赖库，选择用gcc来编译是一件极其繁琐的事情，所以出现了make工具。

**make**
make工具可以看出是一个批处理的工具，其本身没有编译和链接的功能，其的真正功能就是调用已经写好的makefile文件来对项目进行编译和链接。

**makefile**
make工具本身只是一段可执行的程序，而正真要使用这个程序还需要为其输入指令，而输入给make工具的指令就是makefile。我们可以通过编写makefile并输入至make工具，从而使make工具能够根据我们写的makefile文件来定制化的对项目进行编译和链接。

**cmake**
如果只用make工具，通过定制化的makefile文件已经可以解决中小项目的编译问题了，但是面对大型的C/C++项目，使用编写makefile也是极其繁琐的，并且还要面临着不同平台的makefile无法保持一致的跨平台问题，所以便出现了cmake。cmake能够通过为其输入指令（即CMakeList.txt）来让其对应的生成makefile文件，从而供make工具使用，从而达到对项目编译的目的。cmake根据不同的平台产生不同的makefile文件，但是在不同的平台中，make能够根据不同的makefile文件产生相同的效果，由此解决了跨平台的问题。

**CMakeList.txt**
将make看出一段可执行的程序，那么makefile就是该程序的输入；同样，将cmake看出一段可执行的程序，那么CMakeList.txt就是cmake工具的输入。

综上所述，如果你最终要用cmake编译一个项目，只需要对应填写CMakeList.txt文件即可。

### makefile快速入门

先要一文件夹包含三个文件`a_funtion.cpp`、`a_function.h`和`main.cpp`，内容如下，现要将这些文件编译链接成一可执行文件

```c
// main.cpp 
#include <iostream>
#include "a_function.h"

int main() {
    std::cout << __sum(1, 2) << std::endl;
    return 0;
}

// a_function.cpp 
#include "a_function.h"

int __sum(const int &lhs, const int &rhs) {
    return lhs + rhs;
}

// a_function.h
#pragma once

int __sum(const int &lhs, const int &rhs);
```

**使用g++同时翻译和链接**

再命令行中输入以下语句
```shell
g++ a_function.cpp main.cpp -o main.exe
```

即可得到`main.exe`文件，即直接完成编译和链接 
```shell
g++ *.cpp -o main.exe
```

该命令与第一个命令效果相同，其中`*.cpp`表示所有以`.cpp`后缀的文件，即我们要编译链接的文件。

**使用g++先翻译再链接**

使用g++同时翻译和链接的坏处就是一次编译需要所有文件一起编译，但实际中往往只修改一两个文件，如此便导致了一次性全部编译效率很低。

课是先进行翻译，输出成`.o`后缀文件，再将其进行链接形成可执行文件，如下
```shell
g++ main.cpp -c
g++ a_funtion.cpp -c
```

如此得到了`main.o`和`a_funtion.o`两个只进行了翻译的文件，再通过以下语句进行链接
```shell
g++ *.o -o main.exe
```

即可最终生成可执行文件。

这样做的好处是每次只需要重新翻译修改过的文件，而没修改过的文件`.o`文件保持不变，如此在经过单独的链接操作即可生成可执行文件。

**使用makefile编译1**
新建一个makefile文件，将其输入给make工具，从而使make工具调用g++来进行工程文件的编译。

建立的makefile文件如下
```makefile
main.exe: main.cpp a_function.cpp
	g++ -o main.exe main.cpp a_function.cpp 
```

其中最上面一行的`main.exe`表示输出执行文件的名字，冒号后面的是所要使用的源文件，第二行需要先进行缩进，表示其属于`main.exe`同一工程的内容，然后就直接运行缩进后的指令`g++ -o main.exe main.cpp a_function.cpp `。

可以将以上指令总结为
```makefile
TARGET_NAME: DEPENDENT_FILE
	COMMAND
```

**使用makefile编译2**
上面的方式实际上就是调用了`g++ -o main.exe main.cpp a_function.cpp `这一行指令，实际上可以将指令给变量化，如下

```makefile
CXX = g++
TARGET = main.exe
OBJ = main.o a_function.o

$(TARGET): $(OBJ)
	$(CXX) -o $(TARGET) $(OBJ)

main.o: main.cpp
	$(CXX) -c main.cpp

a_function.o: a_function.cpp
	$(CXX) -c a_function.cpp
```

首先进行了三个变量的定义，分别是`CXX`表示编译指令`g++`、`TARGET`表示输出的文件名`main.exe`和所依赖的对象`OBJ`，然后就开始根据输出的文件不断地来递归到已有的文件。

make会自动检查某个文件是否有修改，从而单独的编译修改的文件。

**使用makefile编译3**
上述的makefile文件还可以简化，如下

```makefile
CXX = g++
TARGET = main.exe
OBJ = main.o a_function.o

CXXFLAGS = -c -Wall

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< 
	
.PHONY: clean
clean:
	erase *.o $(TARGET) 
```

该makefile与上一段makefile不同的地方在于对于`.o`文件处理的写法不同的，该写法更具有普适性，即只要OBJ中有的文件都会进行操作。

添加的代码使得命令`make clean`可以删除`.o`和`.exe`文件，只保留源文件。

**使用makefile编译4**
上一个程序再要新增或减少文件时会出现麻烦,可以将程序改为如下

```makefile
CXX = g++
TARGET = main.exe
SRC = $(wildcard *.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))

CXXFLAGS = -c -Wall

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	erase *.o $(TARGET) 
```

这里新增了一个变量`SRC = $(wildcard *.cpp)`，该语句表示将所有的`.cpp`后缀的文件赋值给`SRC`变量，由此再通过`OBJ`调用该变量即可解决文件增减的问题。

### cmake入门到入魂

**构建一个小项目**
先创建一个`CMakeLists.txt`文件，然后再该文档中写入

```cmake
cmake_minimum_required(VERSION 3.23)

project(main)

add_executable(main main.cpp a_function.h a_function.cpp)
```

`cmake_minimum_required` 指定使用 cmake的最低版本号，`project` 指定项目名称，`add_executable` 用来生成可执行文件，需要指定生成可执行文件的名称和相关源文件。

注意这里的名字不用带后缀，而makefile里的名字要带后缀`.exe`（windows操作系统下）。

还是使用makefile的程序文件，现在档期按文件夹新建文件`build`，然后进入文件夹开始cmake构建，指令如下

```shell
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
```

其中`-G`表示选择编译器的类型，而`MinGW Makefiles`就代表着gcc（windows移植版本），只要第一次构建时使用，默认情况下时使用MSVC来编译的。

上述的命令的执行只能产生makefile文件，而不能产生可执行文件，所以还需要添加一行
```shell
cmake --build .
```

只能使用这个指令在当前文件中（build）产生可执行文件。

**优化这个小项目**
使用`SRC_LIST`和`PROJECT_NAME`
可以用`PROJECT_NAME`在`add_executable()`中来代替工程的名字，可以使用`set()`来为`SRC_LIST`变量以多个源文件名字赋值，如此优化后的代码如下

```cmake
cmake_minimum_required(VERSION 3.23)

project(main)

set(SRC_LIST main.cpp a_function.h a_function.cpp)
add_executable(${PROJECT_NAME} ${SRC_LIST})
```

## 谈谈如何在clion中优雅的写stm32

### 环境配置

clion是按照cmake组织的方式来编译的，而其底层使用MinGW的环境，并且使用编译器arm-none-eabi-gcc，即GNU专为嵌入式开发提供的编译器，最后通过open ocd来进行下载烧录的，所以上述流程归纳如下

- 配置MinGW环境；
- 配置arm-none-eabi-gcc；
- 配置open ocd。

如下一张图清除展示了clion中组织编译的方式
![image-20230416102338429](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230416102338429.png)

以下clion中的设置图也表明了这一关系，其中cmake使用clion内置
![image-20230416102632538](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230416102632538.png)

记得上述工具皆加入环境变量，具体的环境变量路径查询网上教程。

其中cmake工具使用clion自带的，而make使用MinGW环境下的mingw32-make.exe，最后C/C++的编译器为arm-none-eabi-gcc/g++.exe。

### 创建工程

*（2023-6月以前，已更新，但仍可以按照这边的步骤）*

*（该方法不建议使用）*可以直接在clion中创建新工程的选项中选择CubeMX工程，等待一段时间后，其会自动生成一个默认的配置的芯片，打开CubeMX重新按照正常配置即可，注意在Project Manager中选择STM32CubeIDE选项，以及**一定要打开generate under root**（打开了之后才会被clion所识别，并且生成对应cmake，不打开的话会生成一个STM32CubeIDE的文件，由此clion无法识别）。

在配置完成并生成代码后回到clion，clion会让我们选择`.cfg`后缀的文件，这是用于配置open ocd程序烧录调试的配置而使用的，可以在上面预制的文件中选择最接近当前所使用单片机，或者先跳过后面自己重新写。

*（建议使用该方法）*还有一种方法，就是利用以前的工程文件的`.ioc`文件生成，新建一个文件夹，将`.ioc`文件复制到文件夹中，然后通过clion打开这个`.ioc`文件（通过项目工程打开），然后再打开MX生成代码，并且同样**打开generate under root**项即可，若要在后米娜使clion生成mx的cmake文件，可以右键`.ioc`文件，并选择对应的选项。

现在按下编译即可进行烧录，记得时常进行cmake的更新。

每当工程中的cmake有问题或者消失，重新用MX生成一下代码，clion就会自动再生成cmake文件。

综上所述

- 复制对应的`.ioc`文件到对应工程文件夹下；
- 通过clion打开该`.ioc`文件；
- 再在clion中使用MX打开这个`.ioc`文件；
- 在MX中记得**打开generate under root**以及使用模板将`CoreCpp`文件夹的文件添加到生成的代码中（只有第一次创建工程，其余不要添加，不然会导致覆盖）；
- clion自动更新代码和对应的cmake；
- 更新cmake中的头文件路径和源文件路径；
  ```cmake
  include_directories(${includes} CoreCpp/Inc Drivers2/Inc)
  
  file(GLOB_RECURSE SOURCES ${sources} "CoreCpp/*.*" "Drivers2/*.*")
  ```

- 在`main.c`中包含`main_cpp.h`头文件以及添加`HAL_RCC_DeInit()`函数和`main_cpp()`函数；
- 在`main_cpp.h`文件中头文件包含`main.c`的头文件；
- 最后设置`.cfg`文件即可进行编程烧录。

步骤真tm的多。

**创建工程**

*（更新于2023-6-27）*

首先介绍模板文件，图片如下
![image-20230627195425715](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230627195425715.png)

首先要使用的是`rename_it.ioc`，将其复制到工程目录下，并重新命名为工程文件后，用clion打开。

直接在clion内部打开CubeMX，在CubeMX中一定要注意模板文件夹的勾选位置要选择默认，如下

![image-20230627194425997](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230627194425997.png)

点击generate code后，MX即会自动帮我们生成代码，此时clion会识别MX生成的代码，而由clion再生成一些配置文件，包括cmake等等。

上述过程之后，clion会提示你选择`.cfg`文件，该文件用于烧录的配置，选择第一个`stm32h7x3i_eval.cfg`即可。

最后将模板文件夹得另外五个文件同一拷贝到工程目录，选择替换，即可完成工程的创建，此时的工程默认例程是使用串口重定向打印”hello, word“字样。

```c++
  ._dma_buffer (NOLOAD):
  {
    . = ALIGN(4);
    *(.DMA_buffer)
    *(.DMA_buffer*)
    . = ALIGN(4);
  } >RAM_D1
```

上面这串代码加在`STM32H743IITX_FLASH.ld`中165行左右添加。

```c++
__attribute__((section ("._dma_buffer"))) volatile uint16_t adc_value[1024];
```

上面这行代码加在`main_cpp.cpp`函数中，表示定义一个数组。

### 在CubeMX使用模板

**自带MX内部的模板**
CubeMX支持在设置过程中添加模板，以达到在工程文件中添加固定代码的功能，参考链接如下
[【CSDN配置CubeMX模板】](https://blog.csdn.net/qq_25820969/article/details/125327504)

模板还支持自带文件夹，所以可以自己定义文件夹放进模板文件夹中，从而生成代码时自动生成相应文件夹，但是再clion中没有自动将该文件加入cmake路径，只有第一次创建工程，其余不要添加，不然会导致覆盖。

如果要在cmake文件更新后不恢复初始化，则添加的cmake文件在`CMakeLists_template.txt`中设置。

### 记一个f407在时钟树初始化失败的bug

open ocd在使用`f4xx.cfg`文件配置下载时，会默认使用stm32的HSI时钟，而f407的HSI时钟才8Mhz，而ST-LINK最大的速率不能超过系统时钟的八分之一，所以下载时使用HSI必然会导致速度无法跑满，所以`f4xx.cfg`文件会自动将f407设置为PLL状态，从而提高时钟频率，提高下载速率。但是f407的用户手册规定，若当前f407处于PLL状态，则此时无法再改变该PLL的参数，此时表现为下载程序后，会在函数`SystemClock_Config()`中陷入死循环（`Error_Handler()`函数），所以需要在重新配置PLL之前使f407处于默认的HSI状态，故添加函数`HAL_RCC_DeInit()`使得f407处于HSI状态。

**顺便也记录以下困扰我两三天的h743卡在`SystemClock_Config()`函数的bug**
由于有了前面f407的踩坑，当我见到h743也会卡在`SystemClock_Config()`函数后，自然而然的想到也是因为时钟配置的问题（就像f407一样），但是我调试了两天，仍然发现程序会卡在时钟树配置的函数，最后发现只要涉及到HSE相关的配置，不管在clion中用open ocd下载还是使用CUBEIDE，都会最终可在`SystemClock_Config()`函数中，所以最终排除了是由于open ocd的原因导致的（为什么最开始没有使用CUBEIDE来发现该问题也会在这里出现呢，其实我试过了，只不过当时手滑在CUBEIDE中选择了HSI为时钟源的PLL配置，不涉及到HSE，才导致我误以为CUBEIDE没有问题，从而一直在怀疑open ocd的原因），最终发现是晶振的输入引脚虚焊了，将该引脚加固，问题解决。**所以综上，外部晶振不起振会造成程序卡在`SystemClock_Config()`函数。**

### clion中添加dsp库
首先打开硬件浮点单元，clion中直接帮我们预写好了`CMakeLists.txt`，可以在`CMakeLists_template.txt`中修改其中的几行

```cmake
# Uncomment for hardware floating point for f4
add_compile_definitions(ARM_MATH_CM4;ARM_MATH_MATRIX_CHECK;ARM_MATH_ROUNDING)	# 可根据不同的核改变
add_compile_options(-mfloat-abi=hard -mfpu=fpv4-sp-d16)	# 根据不同的核支持的fpu而改变
add_link_options(-mfloat-abi=hard -mfpu=fpv4-sp-d16)
```

```cmake
#Uncomment for hardware floating point for h7
add_compile_definitions(ARM_MATH_CM7;ARM_MATH_MATRIX_CHECK;ARM_MATH_ROUNDING)
add_compile_options(-mfloat-abi=hard -mfpu=fpv5-d16)
add_link_options(-mfloat-abi=hard -mfpu=fpv5-d16)
```

将接下来的三行取消注释，即可开启FPU。

然后导入DSP库，可以在CubeMX中直接导入，此处神略不写，具体看链接。

再然后，在`CMakeLists.txt`中链接该库，即在`CMakeLists_template.txt`中添加一行代码

```cmake
add_executable($${PROJECT_NAME}.elf $${SOURCES} $${LINKER_SCRIPT})

target_link_libraries(${PROJECT_NAME}.elf ${CMAKE_SOURCE_DIR}/Middlewares/ST/ARM/DSP/Lib/libarm_cortexM4lf_math.a)	# for f4

target_link_libraries(${PROJECT_NAME}.elf ${CMAKE_SOURCE_DIR}/Middlewares/ST/ARM/DSP/Lib/libarm_cortexM7lfdp_math.a)	# for h7
```

上述第一行是已经存在的，下面的才是需要添加的。

最后就可以包含`#include "arm_math.h"`从而调用DSP库，注意一定要在`#include "main.h"`之后包含DSP库。

**原理**
使用MX添加DSP库的原理为，先添加头文件`.h`和库文件`.a`，这步可以自己完成，也可以使用MX完成。使用MX完成的步骤参考上述网站。然后就是在cmake中添加参数，其中包含添加宏定义和编译链接选项。宏定义由cmake模板自动生成，只需要取消注释即可；编译选项需要自己添加，具体参见上述代码。

[clion中添加dsp库]: https://www.bilibili.com/read/cv19024271/

### clion中添加Eigen库

首先下载获得Eigen库，下图是文件列表
![image-20230628123034646](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230628123034646.png)

我们只需要使用Eigen文件夹。将该文件夹包含头文件中，即可完成添加。

因为Eigen里的所有定义都是在头文件中编写，所以整个库包括定义都属于头文件的内容，所以只需要放在头文件中包含的库就行了，为了方便，我将Eigen文件夹整个放在stm32工程的`Middlewares\Inc`目录下。

## 巨硬Visual Studio的使用

一个solution可以包含多个project，对于小型工程，将solution和project放在同一文件夹中即可。

## VIM使用

VIM根据不同的模式进行使用，直接在输入命令`vim`会进入一个没有文件名的文件中，而在命令`vim`后添加文件名（即使原来目录没有）可以新建或打开一个以当前文件名命名的文件，此时映入眼帘的就是正常模式；

在正常模式下敲击键盘`i`可以自动进入插入模式，此时左下角自动对应为`--INSERT--`，表示插入模式，在插入模式下，就可以任意敲击代码。再插入模式下点`ESC`，则退回正常模式。

在正常模式中，可以使用方向键进行光标的移动，敲击`hljk`有同样的效果，分别对应左右上下。
不仅如此，在正常模式中敲击数字`0`表示硬回到当前行行首，敲击`$`则硬到当前行行尾，在前面命令的基础上加入数字`n`，可以表示下n行的行首或行尾，如`30`和`4$`分别表示下3行的行首和下4行的行尾；既然有了硬行首，那么也一定有软行首，敲击`^`则会到达当前行的软行首。
也可以在正常模式中的命令`+`和`-`可以表示向上和向下一行的软行首（忽略空格），前面加数字表示重复该命令数字值次。


在正常模式中敲击大写的`ZZ`表示保存并退出；敲击`ZQ`表示强制退出，此时不会保存本次敲入的字符。

正常模式中敲击冒号`:`表示进入命令模式，其中`:w`表示保存，若此文件没有文件名，会无法保存。此时需要再添加文件名来进行保存`:w filename`，这样就会在当前的文件夹中出现一个文件名字为`filename`，如果已经有了名字仍然使用`:w filename2`，那么则会再在当前文件夹中产生新的文件名字为`filename2`，而此时的vim打开的是新的`filename2`文件，旧的`filename`文件则会保留在文件夹中。

在命令模式种使用`:qa`可以退出vim，但是若当前未保存，则vim会询问你是否要真的退出，所以该指令只有在保存了后才能迅速退出。所以可以使用`:wqa`来同时进行保存和退出操作。

vim中有以隐藏文件`.vimrc`（linux中隐藏文件都以`.`开头，可以使用`ls -a`来展示隐藏文件），该文件的作用是在每次打开vim前会自动执行里面的指令，所以如果有一些指令需要每次vim中都使用，那么可以在`.vimrc`中写入该指令，如写入`:set mouse=a`表示开启鼠标光标的移动；写入`:set nu`表示开启行号。

在正常模式种使用`w`命令，可以一次跳转一个单词，而使用大写`W`（也就是`shift + w`）就是增强版的`w`，一次可以跳过一些其他符号。
与`w`相反的命令为`b`，即向左跳跃一个单词，同样也有增强版的`B`（即`shift + b`），增强版的跳跃单词更好用的是使用`control + left/right`。
使用`e`和大写`E`，跳转至每个单词的末位（上述都是开头）。

在正常模式中敲击`x`直接删除当前光标的字符。

正常模式中敲击`v`进入虚拟模式`--VISUAL--`，此时的移动都会视为选中，常常在利用快速跳转的命令（例如`w`等）选择后配合`x`来在正常模式中进行快速删除操作。

在正常模式中删除同样可以敲击`d`，与`x`不同的是，`d`会继续等待你输入其他按键，如`de`表示当前光标到该单词的末位进行删除，等价于`vex`；或`d$`表示删除当前行到硬句首，等价于`v$x`，总之`d`接着下一步跳转操作，把在这之间所有选中光标都进行删除。同样的大写的`D`表示直接删除当前光标之后的所有内容，等价于`d$`。

若是要更改当前的某一个单词，按照之前的方法可以`dwi`，即先将当前单词删除，然后进入插入模式，这样做未免有点繁琐，可以使用`cw`来简化上述操作，`c`同样表示删除，但是删除之后立马进入插入模式。

使用`g`相关的指令可以在正常模式下快速跳转至某一行，如跳转至55行，则需要敲击`55G`或者`55gg`，需要跳转到第一行，则敲击`gg`，到达最后一行，则敲击`G`。

可以在正常模式中快速查找某个单词，只需要使用命令`/wordname`，然后光标就会立刻跳转到第一个出现该单词的位置，紧接着使用`n`则会跳转到下一个该单词出现的地方，`N`是跳转上一个单词的位置。

## 不知道哪来的BUG

### 关于串口重定向会影响ADC的DMA功能这件事

在h743中使用adc的DMA模式，用定时器触发。在调试的过程中，发现只要使用了串口重定向，即使用了`stdio.h`内的函数，就会导致传输的数据不正常，不知道为什么。

还有，当传输的数据长度过少时，会导致数组全是0，我也不知道为什么。

最后解决问题的方式是改变数组的内存位置，在文件`STM32H743IITX_FLASH.ld`添加这行代码

```c
  ._dma_buffer (NOLOAD):
  {
    . = ALIGN(4);
    *(._dma_buffer)
    *(._dma_buffer*)
    . = ALIGN(4);
  } >RAM_D1
```

然后声明变量的时候使用
```c
__attribute__((section ("._dma_buffer"))) volatile uint16_t adc_value[SAMPLE];
```

