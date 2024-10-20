## Altera with Quartus

### Create a Project with Quatus

**page 1 of 5**

此页配置主要选择工程文件夹中的prj位置和名字（要与总工程文件相同）

- File -> New -> New Quartus 2 Project。

- 在工程文件（名字记住）位置创建四个文件夹（doc、prj、rtl、sim）。
- What is the working directory for this project？中选择prj文件。
- What is the name of this project？中选择整个工程文件的名字，若名字不一样，记得后面将未见设置为顶层即可。

**page 2 of 5**

添加代码文件，代码文件的名字必须和**代码内部module的名字**相同，代码文件一般自己创建在rtl中。代码文件也可在后面添加，记得点add。

**page 3 of 5**

![image-20221021210340996](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20221021210340996.png)

选择对应芯片的型号。

**烧录文件**

simulation选项改为Modelsim。

然后写代码。

绑定管脚 Pin Planner。

引脚默认电位设置 Assignments -> Device -> Device and Pin OptionS -> Unused Pins -> As input tri-stated

写完代码先验证对不对 Start Analysis & Synthesis

之后 Start compilation生成文件

打开Programmer -> add file 组后开始start转换文件

**testbench编写**
testbench代码文件的原理是modelsim运行该代码，然后捕捉改代码中的内部变量，所以testbench不需要输入输出端口。

```verilog
`timescale 1ns/1ns
module tb_divider ();
	reg		clk;
	reg		rst_n;
	
	wire	clk_out;
	
	initial begin
		clk = 1'b1;
		rst_n = 1'b0;
		#20
		rst_n = 1'b1;
	end
	
	always #10 clk = ~clk;
	
	calculagraph calculagraph_inst (
		.clk_ori(clk), 
		.rst_n(rst_n), 
		
		.clk(clk_out)
	);
	
endmodule
```

分为三部分

- 时基、精度、模块和内部变量，其中时基、精度和模块十分固定，注意模块名和文件名相同
  ```verilog
  `timescale 1ns/1ns
  module tb_divider ();
  ```

  然后就是内部变量，需要先声明被仿真文件的输入输出端口，其中输入对应的变量为`reg`类型，输出对应的端口必须为`wire`类型。

- 其次就是输入的仿真，一般为`clk`和`rst_n`，并且模式也是固定的，如上代码所示，将`rst_n`置低，后20ns后置高模拟上电复位。
  然后单独一个`always #10 clk = ~clk;`来模拟`clk`的产生。

- 最后一部分实例化，模式也固定，把第一部分声明的内部变量对应上就行。

### The Tool Flow

coding

simulation

synthesis 将描述性语言转化为门电路

compile 烧录进fpga

### In and Out

```verilog
module simple_in_n_out (
	input	in_1, 
	input	in_2, 
	input	in_3, 
    
	output	out_1, 
    output	out_2
);
	// Port definitions
	
	
	// Design implementation
	assign 	out_1 = in_1 & in_2 & in_3;
    assign 	out_2 = in_1 | in_2 | in_3;

endmodule
```

该例展示了一个最简单的组合逻辑电路设计，其模块只声明了最基本的模块输入和输出，并且直接通过输入来得到输出。

该电路包含了语言的三个部分

- 运用module语句说明模块的名称（工程名，文件名，顶层模块名要一致）和展示了端口列表，即输入输出端口。
- 端口声明，旧版本的端口列表需要单独声明，而现在可以集成在module语句中，除此之外还有一些其他中间变量的声明。
- 实际运行，如上述运用assign对output端口进行赋值从而实现组合语句。

```verilog
module intermed_wire (
	input	[3 : 0]		in_1, 
	input	[3 : 0]		in_2, 
	input				in_3, 
    
	output	[3 : 0]		out_1
);
	// Port definitions
	wire	intermediate_sig;
	
	// Design implementation
    assign	intermediate_sig = in_1 & in_2;
    
	assign 	out_1 = intermediate_sig & in_3;
    assign 	out_1 = intermediate_sig | in_3;

endmodule
```

该例在前一个例子的基础上增加了中间变量，即引入了**wire**类型变量，如名字所见，你可以将其理解为与线直接相连，从而充当中间变量。

```verilog
module bus_sigs (
	input	[3 : 0]		in_1, 
	input	[3 : 0]		in_2, 
	input				in_3, 
    
	output	[3 : 0]		out_1
);
	// Port definitions
    wire	[3 : 0]		in_3_bus;
	
	// Design implementation
    assign 	in_3_bus = {4{in_3}};
    assign 	(in_1 & ~in_3_bus) | (in_2 & ~in_3_bus);

endmodule
```

verilog中还有向量的概念，就是多个位组合在一起（单独一个位的变量称之为标量），类似于C语言的数组。

一般的逻辑操作符可以对拥有相同大小的向量进行操作。

复制操作符，如上例的`{4{in_3}}`，由两对花括号构成，数字代表重复的次数，而内部花括号所包含的就是被重复的变量，最终得到的是一个向量。

```verilog
module standard_mux (
	input	[3 : 0]		in_1, 
	input	[3 : 0]		in_2, 
	input				in_3, 
    
	output	[3 : 0]		out_1
);
	// Port definitions
	
	
	// Design implementation
	assign 	out_1 = in_3 ? in_1 : in_2;

endmodule
```

verilog中的条件语句，该语句与C语言中的符合条件语句完全相同，在门电路中表现为数据选择器MUX。

```verilog
module bus_breakout (
	input	[3 : 0]		in_1, 
	input	[3 : 0]		in_2,  
	output	[5 : 0]		out_1
);
	// Port definitions
	
	
	// Design implementation
	assign 	out_1 = {	// pay attention to the order!
						in_2[3 : 2], 
						(in_1[3] & in_2[1]), 
						(in_1[2] & in_2[0]), 
						in_1[1 : 0], 
					};

endmodule
```

将向量拆开赋值的操作称为concatenation，效果如上所见，注意顺序由大下标开始，到小下标结束。

### Clock and Registers

```verilog
module simple_dflop (
	input		clk, 
	input		in_1,
	output	reg	out_1
);
	// Port definitions
	
	// Design implementation
	always @(posedge clk) begin
		out_1 <= in_1;
	end

endmodule
```

由这个例子进入时序逻辑电路的设计。

首先是添加了reg变量，在verilog中，reg是和wire相对的变量类型，后者只能应用于组合逻辑电路（即assign语句），而在时许电路（即always块），只能使用reg类型的变量。
p.s.对于端口列表的输入输出端口，若不指明为reg类型，则其默认为wire类型，并且一般只对输出或输出有关的中间变量用作reg类型。

其次是对于非阻塞赋值`<=`的使用，在verilog中，阻塞赋值和非阻塞赋值仍然是相对的赋值方式，具体内涵不必多说，需要注意的是组合逻辑的assign语句只能使用阻塞赋值`=`，而时许逻辑的always块中既可以使用阻塞`=`也可以使用非阻塞`<=`，但是不会有人选择阻塞赋值吧。

细想也是，非阻塞赋值就是所有块同时以上一时刻的值来赋值，而组合逻辑根本没有并行这一说，也就不能使用非阻塞赋值。

最后是对于always块的使用
```verilog
always @(posedge clk) begin
    out_1 <= in_1;
end
```

always可以由两个部分组成，其一是敏感列表（sensitivity list），其二便是由begin和end所包含在内的具体语句。
只有当敏感列表内的条件触发，才会执行接下俩的语句块，不要被always的表象所迷惑以为他是一个死循环，可以将敏感列表理解为中断而语句块为中断回调函数，只有当中断发送（即敏感列表触发），才会引起终端回调函数的发生（代码块的执行）。

```verilog
module dflop_n_reset (
	input		clk, 
	input		reset, 
	input		in_1,
	output	reg	out_1
);
	// Port definitions
	
	// Design implementation
	always @(posedge clk or posedge reset) begin
        if (reset)
            out_1 <= 0;
        else
            out_1 <= in_1;
	end

endmodule
```

要实现异步重置的功能，只需要在always的敏感列表中添加新的敏感条件`posedge reset`即可，接着在代码块中使用if/else语句，如上面代码所示。

```verilog
module dflop_en_clr (
	input		clk, 
	input		reset, 
	input		clear, 
	input		enable, 
	input		in_1,
	output	reg	out_1
);
	// Port definitions
	
	// Design implementation
	always @(posedge clk or posedge reset) begin
		if (reset)
			out_1 <= 0;
		else if (!clear)	// clear high synchronous set 0
			out_1 <= 0;
		else if (enable)
			out_1 <= in_1;
	end

endmodule
```

添加两个同步功能，clear和enable，当clear为0时同步置0，enable要一直保持1才可以正常工作。

为了保持一致性，请为所有时许电路都添加上同步和异步的置0的功能。

与所有其他的编程语言一样，if/else if语句用于多选一，且前面优先级越大，并且也启示我们，当遇见有不同的条件对应的功能不尽重合，选择if/else if语句，并且按照优先级排列，最后在简化或合并条件从而简化代码。

```verilog
module srflop_n_cntr (
	input				clk, 
	input				reset, 
	input				start, 
	input				stop, 
	output	reg	[3 : 0]	cnt
);
	// Port definitions
	reg			cnt_en;
	reg 		start_d1;
	reg 		start_d2;
	// Design implementation
	// Enable counter, manage cnt_en
	always @(posedge clk or posedge reset) begin
		if (reset)
			cnt_en <= 1'b0;
		else if (start_d2)
			cnt_en <= 1'b1;
		else if (stop)
			cnt_en <= 1'b0;
	end
	
	// Counter
	always @(posedge clk or posedge reset) begin
		if (reset)
			cnt <= 4'h0;
		else if (cnt_en && cnt == 4'd13)
			cnt <= 4'h0;
		else if (cnt_en)
			cnt <= cnt + 1;
	end
	
	// Start delay
	always @(posedge clk or posedge reset) begin
		if (reset) begin
			start_d1 <= 1'b0;
			start_d2 <= 1'b0;
		end
		else begin
			start_d1 <= start;
			start_d2 <= start_d1;
		end	
	end

endmodule
```

此例程实现了一个模13（4位宽）计数器，其中start延迟两个clk后计数器开启，stop表示停止。

该程序使用了三个always块，每个块对应一个简单的功能，如第一个块负责使能计数器，管理start和stop的输入；第二个块负责计数，在保持cnt_en为1的同时循环计数；第三个块负责实现start的延迟开启功能，真正的开启信号为start_d2，而start和start_d1都为中间信号。

第一个always块通过检测start和stop的标志来决定是否开启计数，即使能cnt_en。
第二个always块则通过检查前一个块的cnt_en来决定是否开启计数，当且只有cnt为1时才会正常计数，并且到终止返回0。
第三个always块则作为延迟功能，通过start_d2来联系第一个always块。

最后的延迟几个周期的代码实现可以记住。

verilog一般不使用嵌套条件语句，所以要实现类似的功能时，使用并行的if-else语句，并且每个if的第一个条件为原来嵌套语句的最外层条件，如下
```verilog
if (cnt_en) begin
    if (cnt == 4'd13) begin
        cnt <= 4'h0;
    end
    else begin
        cnt <= cnt + 1;
    end
end
// 等同于以下语句
if (cnt_en && cnt == 4'd13) begin
    cnt <= 4'h0;
end
else if (cnt_en) begin
    cnt <= cnt + 1;
end
```



### State Machines

这里给出一个状态机的例子，状态转移图如下
![image-20221113122812886](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20221113122812886.png)

代码如下
```verilog
module state_machine_1 (
	input		clk, 
	input		rst_n, 
	input		go, 
	input		kill, 
	output	reg	done
);

	// State definitions
	parameter	idle = 2'b00;
	parameter	active = 2'b01;
	parameter	abort = 2'b10; 
	parameter	finish = 2'b11;
	
	// Port definitions
	reg	[6 : 0]	cnt;
	reg	[1 : 0]	state_reg;
	
	// State machines
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			state_reg <= idle;
		end
		else begin
			case (state_reg)
			idle:
				if (go) begin
					state_reg <= active;
				end
			active:
				if (kill) begin
					state_reg <= abort;
				end
				else if (cnt == 7'd99) begin
					state_reg <= finish;
				end
			finish:
				state_reg <= idle;
			abort:
				if (!kill) begin
					state_reg <= idle;
				end
			default:
				state_reg <= idle;
			endcase
		end
	end
	
	// Counter when active
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			cnt <= 7'b0;
		end
        else if (state_reg == finish || 		// 先写归零，再写递增
            	 state_reg == abort) begin
           cnt <= 7'b0; 
        end
		else if (state_reg == active) begin
			cnt <= cnt + 1;
		end
	end
	
	// Done register when finish
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			done <= 1'b0;
		end
		else if (state_reg == finish) begin
			done <= 1'b1;
		end
		else begin
			done <= 1'b0;
		end
	end
	
endmodule
```

状态的分配使用可以用两种语句，其一是用parameter，可以将其理解为声明常量，在该模块中使用不会再更改，声明的时候不需要指定位宽；另外一个是define，以后会涉及。

状态机的状态转移是使用case语句，语法如下
```verilog
case (state_reg)
idle: // some statements
active: 
    if (state_reg == kill) // ...
// some states
default:
endcase
```

再某个case的情况下用if语句来进行状态的转移，大部分时候是涉及state_reg之间的转换，少部分如需要其他变量（如计数器计满）满足某个条件触发状态转移。

配合着状态机的always，需要为状态机的每个状态编写对应的always块，其触发的条件就是state_reg，如cnt的递增操作，而是否计满则是要交给状态机的always来负责。

### Monitor in Testbench

```verilog
initial begin
    $timeformat(-9, 0, "ns", 6);
    $monitor("@time %t: val: %d seg: %b", $time, dis_val, dis_seg_struc);
end
```

使用系统函数`monitor`来监控被观测的值，只要值发生变化，就在simulation中打印出来。

系统函数`timeformat`用来设置时间单位格式，-9表示1e-9s，0表示小数点后显示0位，6表示`%6d`。

建议单独使用初始化模块来输出这些语句。





任老师教的

- 不写test_bench仿真，直接打开modelsim
- 顶层文件图表连接
- 调用ip核
- 使用逻辑分析仪

modelsim快速操作指令
