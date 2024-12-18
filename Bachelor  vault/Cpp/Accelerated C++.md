*2022.1*

#### 表达式

表达式包含两个概念：操作数和运算符

每个操作数都有一个类型，运算符根据操作数来决定进行的操作。

每一个表达式会产生：结果和副作用。副作用就是对程序造成的影响，结果就是返回值。

运算符有个结合特性：左结合和右结合，由此可以判断某个表达式是从左到右还是从右到左。
        左结合：表示运算符左边可以装有更多的运算符，即可将运算符左边全部用括号括起来（右结合同理）。

例：辨析以下代码

```c++
const std::string exclam = " ! ";
const std::string message = "Hello" + ", world" + exclam;
// 以上代码编译错误
// 以上表达式有两个操作符"="和"+"，前者优先级较低并且为右结合律，后者高一些，为左结合律（并没有高多少）
// 整个表达式先执行+运算符，同时出现两个+运算符，由于左结合，所以前面的+先执行，此时的+运算符的操作数都是字符串字面量，未定义此类型的加法运算，故b
```

**使用auto前置的初始化方式**
使用旧风格的类型前置的初始化方式很容易忘记初始化，从而导致后面出现问题。可以使用`auto`前置的方式来避免，一是`auto`关键字必须得使用初始化表达式来推断其表示的类型，由此你必须进行初始化，如定义一个`char`类型的变量`ch`

```c++
char ch = 0;	// 不推荐，很容易忘记初始值
auto ch = char(0);	// 推荐，必须有个初始值
```

还可以使用lambda表达式来更灵活的初始化变量，如
```c++
auto flag = bool(true);
auto i = [&] {
    if (flag) {
        return char(1);
    }
    else {
        return char(0);
    }
} ();
```

上述函数首先使用类型后置的方法来声明并初始化一个布尔变量`flag`，紧接着使用类型后置配合lambda表达式的方法来控制初始化的值。

使用lambda表达式的机理在于在定义的立刻使用调用运算符来调用该lambda表达式；使用`&`为lambda表达式的捕获列表，由此可以捕获相邻作用域的所有变量；最后返回一个显示的强制类型转换的方式来让`auto`显示的推断类型。

#### decltype类型指示符

根据表达式的类型来得到一个类型来用于声明变量。

值得注意的是：

1. decltype()不可用const之类的关键字来修饰。（单变量）

2. decltype()返回的类型完美继承了括号内变量的修饰，包括const或引用。（单变量）

3. 若括号内为表达式，则以表达式返回的类型（有些表达式返回右值，有些表达式返回左值）
   ```c++
   int i = 5, &r = i;
   decltype(r) a;	// 错误，r是引用值
   decltype(r + 0) b;	// 正确，为表达式的类型。
   decltype((i)) c;	// 错误，(i)当作表达式返回的是左值
   ```

   

#### 优先级、结合律和求值顺序的区别

将每一个操作符都当作一个函数，而操作符的操作对象当作函数时参数，而操作符的优先级和结合律就决定了函数参数应该为表达式中的哪一个部分。

优先级具体规定的是当两个级别不同的操作符相邻时，哪一个操作符先进行的问题。

结合律具体规定的是当两个或多个优先级级别相同的操作符相邻时，具体的组合方式。
```c++
double a = 10 * 20 / 5;
// 等价于
double b = (10 * 20) / 5;	/* '/'的左结合律，故左边括号 */
```

而求值顺序就是规定了变量得到值得顺序。

```c++
a + b * c;	//优先级和结合律规定了最终b的值和c的值先相乘，后在和a相加，而没有告诉你a，b，c得到值的先后顺序是怎么样的。
f() + g() * h() + j();
```

还是那句话，多用括号；

找到最小单元，函数调用算最小单元，最小单元的求值顺序大部分是未定义的。

还是有一些不理解。

```c++
double wage, salary;
wage = salary = 999.99;
```

赋值运算符`=`的结合律为右，所以先往右扩括号。

```c++
wage = salary = 999.99;
wage = (salary = 999.99);
```

优先级和结合律只是充当了一个括号的作用，所以并没有实际的算出值来，求值顺序才真正算出了具体的值。

C++没有明确大多数规定的二元运算符的求值顺序。

#### 循环不变式

为了方便理解循环，讲循环中的判断变量与某个操作（或说是预期）联系在一起。
如何正确写出不变式：

1. 第一次进入判断条件之前是正确的。
2. 每次循环条件结束，即将进入下一次判断条件时也是正确的。

```c++
const int rows = 5;
// 不变式：到此为止，以及输出了r行
r = 0;
// 第一次进入判断时：由于还没有输出，故输出了r=0行，正确。
while (r != rows) {
    // statement
    ++r;
    // 循环结束时：r增加且输出操作进行，故不变式正确。
}
// 两个判断都为真，不变式正确，可以根据不变式充分理解循环语句。
// 不如此例为r!=rows，最终r一定等于rows，所以此时根据不变式，必定输出了r=rows行
```

#### 控制器

```c++
endl; // 在<iostream>，换行。
setprecision(3); // 在<iomanip>，设置后续流精度3。
```

#### 有关向量复制到另一向量末尾的总结

1. 循环，遍历待复制的向量，向原向量末尾`push_back()`
2. 用insert（非泛型）方法，`vec.insert(vec.end(), copy_vec.begin(), copy_vec.end())`
   该顺序容器的方法`insert()`将`[copy_vec.begin(), copy_vec.end())`范围内的向量插入到`vec.end()`的开始（包括这个元素），若`vec.end()`后面有元素，则覆盖。
3. 用copy（泛型），`copy(copy_vec.begin(), copy_vec.end(), back_inserter(vec))`
   其中`back_inserter(vec)`返回一个迭代器，这个迭代器用作参数时会向后添加空间以复制。
   泛型算法`copy`以三个迭代器作为参数，即将范围`[copy_vec.begin(), copy_vec.end())`内的参数复制到以第三个迭代器参数开头的容器中，但是其不包含检测容器大小是否足够，所以违反数量规定会引起未定义行为。
   `back_inserter()`函数由一个顺序容器作为参量，返回其末位的可增加容器大小的迭代器，所以上述使用`copy()`函数时就算超出了容器边界，也会因为`back_inserter()`函数而增加原容器的大小。

重载函数不适合用于需要函数名作为参数的函数，所以需要重新定义

#### 将函数当作对象作为另外一函数的形参

一般是在有大量重复操作且仅有一个十分类似的函数操作中有不同的时候使用，比如分别计算平均值和中值。

#### 泛型编程的例子

```c++
template<class T>
T median(std::vector<T> vec) {
	if (vec.empty()) {
		throw std::domain_error("median of an empty vector.");
	}

	typedef std::vector<T>::size_type vec_sz;
	std::sort(vec.begin(), vec.end());
	vec_sz median_index = vec.size() / 2;
	return vec.size() % 2 == 0 ? (vec[median_index - 1] + vec[median_index]) / 2 : 
								 vec[median_index];
}
```

函数头`template<class T>`告诉编译器以下定义的函数是一个函数模板，其中泛型的参数类型为T，实际上，对模板函数的每一次调用，都会使该模板产生一个实例化，用实际的类型（如int）来实例化该模板，由此编译器会产生一个以int代替T相同效果的函数。

对于`typedef`，由于使用了泛型的参数类型T，编译器无法了解`std::vector<T>::size_type`所表示的涵义是该容器的一个类型，所以需要提前使用`typedef std::vector<T>::size_type vec_sz`来告诉编译器表示的含义为一个类型。

编译器将会从两个角度来判断传入的类型是否能够替代泛型类型T，其一是C++内部的，由于不同的类型所支持的操作不尽相同，所以当传入一个不适当的类型给T时，有可能会在模板执行的过程中遇到该确定类型为定义的操作，从而引发错误；其二是C++外部的，例如STL库中的大多数泛型算法都是基于迭代器来进行的，而迭代器的这一要求就是STL对模板使用者所传参数类型的约束。实际上，泛型模板本身就是基于不同类型中相似的地方进行定义的，只有某一类型拥有该模板支持的所有操作，才是何时的参数类型。



#### algorithm库中一些泛型函数的用法

`find_if(iter_start, iter_end, function_name)`其中循环遍历`[iter_start, iter_end)`直到让`function_name`返回`true`。

`transform`函数：

```c++
transform(contain.begin(), contain.end(), back_inserter(contain2), function_name);
// 将contain内所有的元素对应function_name映射到contain2中
```

`remove_copy(contain.begin(), contain.end(), back_inserter(contain2), a_num)`表示在`contain`中除了`a_num`都会被复制到`contain2`中，且不改变原有向量的成员。（这个函数的字面意思其实也就是remove（删除）`a_num`元素，只不过转移到了另外一个向量，去元素复制）

`remove(contain.begin(), contain.end(), a_num)`与`remove_copy`类似，只不过在同一序列完成删除，将不能删除的元素往前覆盖掉需要删除元素的空间，从而达到删除的效果，返回不被删除的最后一个元素的后一个元素的迭代器。

`stable_partition(contain.begin(), contain.end(), function_name)`划分区域，满足条件划到前面，否则在后面，返回在后面的第一个元素迭代器。（`partition`函数具有相似用法。）

`std::accumulate(vec.begin(), vec.end(), 0)`，该函数在默认情况下，以第三个参数的为初值，进行累加，而累加过程中返回值（中间值）的类型都与初始值的类型相同，也就是说，若是传入容器元素的类型在累加的过程中会溢出，那么则可以通过给定第三个参数不会溢出的类型来防止溢出的发生，注意，该函数不是定义在`<algorithm>`库中，而是`<numeric>`中，为了方便才写在这。

#### 迭代器

标准库中提供了五种类型的迭代器，其中的每种都对应着一种操作集，标准库的每一个算法都会支持这五种迭代器的其中一种，由此可以更加明确标准库算法函数的支持类型。

**输入迭代器（顺序只读访问）**
输入迭代器，只需要迭代器满足顺序（`++`运算符），只读（`*`、`->`运算符），另外附加迭代器之间能够判断大小关系（`==`、`!=`）即可。

像这样的迭代器叫做**输入迭代器(input iterator)**，诸如`std::find`函数所需要的泛型迭代器就是输入迭代器。

**输出迭代器（顺序只写访问）**
将输入迭代器的只读（`*`运算符）更换至只写`*it = val`操作，就将输入迭代器改变成了输出迭代器，模板中命名为`OutputIt`。

除此之外，对于使用输出迭代器的程序来说，其还需要满足**一次写入**的特性，即迭代器的每次递增都有且只能进行一次赋值运算（不进行赋值操作也不行），当然这是对于使用输出迭代器的程序的约束，`std::copy`就是输出迭代器的例子，模板中命名为`OutputIt`。

**正向迭代器（顺序读写访问）**
输入迭代器和输出迭代器相结合，即某一迭代器在可以顺序（`++`）的基础上，又可以进行读（`*`），又可以进行写（`*it = val`）操作，称这样的迭代器叫做**正向迭代器(forward iterator)**。

当然，正向迭代器不再需要满足输出迭代器的一次写入特性，但是由于该迭代器没有`--`操作，所以在读或写后就再也不能访问该元素了，`std::replace`就是这个例子，模板中命名为`ForwardIt`。

**双向迭代器（可逆访问）**
在正向迭代器的基础上，还可以支持`--`操作，即可称为双向迭代器（bidirectional iterator），`std::reverse`中有使用，模板中命名为`BidirIt`。

**随机访问迭代器（随机访问）**
在双向迭代器的基础上支持迭代器的算数运算操作，如`p + n`，`p - n`、`p - q`、`p[n]`(`*(p+n)`)和`p < q`比较。

`std::sort`就用到到了随机访问迭代器，模板中命名为`RandomIt`。

#### 构造函数

构造函数是用于对自定义类的自定义初始化方式进行操作的，其中可包括多种重载，根据不同的参数从而确定是哪种重载函数。

构造函数是普通函数的阉割版本，不含返回值。

唯一需要注意的是：

```,
Student_info: midterm(0), final(0) { }
```

中间是用于初始化的。

#### C++内置的内存管理函数

```c++
p1 = new T;    // 返回一个T类型的空间的指针，进行T的默认初始化
p2 = new T(args);  // 同上，只不过初始化为args
p3 = new T[n];    // 数组的指针
delete p1;
delete[] p3;
```

缺点是：分配的空间是已经初始化的空间，故缺少了很多的灵活性。

#### allocator类内存管理

与C++内置的new和delete不同，new和delete是将分配内存和初始化一起完成的，但是这样就缺乏了灵活性和效率，因为往往会多次的进行初始化和赋值。
因此标准库memory提供了将分配内存和初始化分配的操作，allocate类。用法如下：

| 方法                  | 作用                                                         |
| --------------------- | ------------------------------------------------------------ |
| `std::allocator<T> a` | 实例化一个将要分配内存的对象，对象类型为T                    |
| `p = a.allocate(n)`   | 分配n个T对象大小的内存，内存未初始化，返回首地址的指针。     |
| `a.construct(p, val)` | p是指针，指向某个未初始化的从allocate分配的内存，用val初始化。 |
| `a.deallocate(p, n)`  | 释放内存，但要求内存内没有建立元素，且n必须是分配时的n。     |
| `a.destroy(p)`        | 将p位置的对象摧毁，内存仍然保留。                            |

此外还有两个函数：

| 函数                             | 作用                                           |
| ------------------------------ | -------------------------------------------- |
| `uninitialized_copy(b, e, b2)` | 用[b, e)的对象初始化b2这个未初始化刚分配的内存，返回刚刚初始化的最后一个迭代器。 |
| `uninitialized_fill(b, e, t)`  | 在为建立对象的[b, e)中以初始化t，返回刚刚初始化的最后一个迭代器（同上）。     |

#### 三位一体规则(rule of three)

需要为类定义**初始化**（包含初始化时的**复制**）、**赋值**、**删除**等操作，就涉及到了**构造函数**（其中包含**复制构造函数**）、**重载运算符函数**、**析构函数**。

| 操作      | 语法                                                                                                                                                         |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 构造函数    | 在类中的声明为与类同名的函数，若是直接在类中定义，最好的方式是<br />`Student_info(): midterm(0), final(0) { }`可以有多个重载，具体方式在花括号内定义。<br />若是在类外定义函数必须在函数名前面加上类的命名空间，其余的就不用了。                |
| 复制构造函数  | 构造函数的一种重载，函数参数为本身类的类型，目的是在初始化的时候可以作为其他实例的副本。                                                                                                               |
| 重载运算符函数 | 需要有返回值，一般为非常量引用`Vec &operator=(const Vec &)`<br />这个函数重载了`=`故要根据原来符号的特性来具体定义函数。<br />在类外定义要加上模板类型`template<class T> Vec<T> Vec<T>::operator=(const Vec &)` |
| 析构函数    | 与类名相同，前面加一个`~`如`~Vec()`无返回值无参数，作用就是删除这个对象里面的内容。                                                                                                            |

析构函数不是用来自己用的，是给编译器使用的，当要释放这个变量的时候编译器应该做什么，如果编写的错误的析构函数，该对象在需要释放的时候编译器无法通过你写的析构函数删除释放，则容易导致内存泄漏。

**默认的”三位一体“**：

若类中没有显示地定义构造函数等”三位一体“规则下的函数，那么编译器会自动地进行默认”三位一体“，就其中地数据成员分别在根据其三位一体来操作（不断递归）。

#### 自定义类型的自动转换

##### 其他类型转换为自定义类型

一般来说赋值表达式的左操作数和有操作数的类型是相同的，尤其是在自定义类中尤其重要，但是如果已经定义了由右操作数类型转换为左操作数类型的**构造函数**，编译器便会自动的将有操作数转换为左操作数的类型。

机制就是他先按照左操作数的类型以右操作数作为参数初始化，因为刚好有以右操作数类型为参数的构造函数所以能够成功的构造做操作类型的变量，故能够实现赋值操作。

```c++
Str str1;
str1 = "hello!";
// 上面等价于
Str temp("hello!"); // 初始化
str1 = temp;
```

##### 自定义类型转换为其他类型

在自定义类中添加类型转换函数

```c++
operator double { return grade(); }        // grade()函数计算出一个double类型并返回
```

进而编译器可以自动转换类型。

请时刻注意如果定义了这种自动转换类型函数，在使用未进行重载的运算符时可能会进行自动转换从而虽然没有重载运算符，任然过得了编译，但却是错误的。

#### 关于重载运算符函数该在类里或者类外定义

首先记住一个准则：**改变操作数本身数据的在类里，不改变的在类外。**

- 比如索引`[]`、`+=`等都会改变数据内部状态，故在类里定义。

- 但是如同`+`等不会改变内部状态，所以在类外定义。

除此之外有一个例外，就是遇到**运算符左右不对称**的情况，左边的变量不为类类型，就必须要类外。

在上述的基础上判断是否要访问类中的私有类型而选择是否友元（针对类外定义）。

还有一点需要注意，成员函数的操作数非对称，即左操作数不能被自动转换到对应类型，而非成员函数具有对此性，所以一般在哪些支持类型转换的重载运算符函数要定义在类外。

#### 友元

##### 友元函数

类外定义的函数能够访问类中的私有数据成员。

仅仅只要在类中补充上函数声明，并在前面加上`friend`关键字。

```c++
class Str {
public:
    friend std::istream &operator>>(std::istream &, Str &);
    // 其他
}

std::istream &operator>>(std::istream &is, Str &str) {
    // 其他
    return is;
}
```

注意函数定义的格式与是否友元无关，但是函数的参数中必须包含类类型。

##### 友元类

与友元函数类似，在某个类中声明友元类后，其他类就可以访问该类的私有数据成员。

```c++
class Core {
public:
    friend class A;
    template<class T> friend class B;
}
```

友元不可以被派生类继承（有待查证）。

#### 自定义string类在转换成字符指针的缺陷

可以在自定义string类中添加转换类型操作函数：`operator char *()`

- 首先不能直接返回私有数据`data`，就算data类型是字符指针，也会破坏类的封装性。
- 如果要开辟一个新的内存来将`data`复制到内存中，返回指向该内存的指针，在显示的时候（如将返回指针赋值给另外一指针）可以，但是如果隐示的操作操作中就会造成内存泄漏（开辟了内存却丢失了这块内存的指针）。

string标准库提供了三种方式实现`string`到`char *`的转换：

| 方法          | 原理                                                           |
| ----------- | ------------------------------------------------------------ |
| `s.c_str()` | 返回一个常量指针以`'\0'`结尾，该指针是string内建的，不能修改，并且会在下次修改string时使原来指针失效。 |
| `s.data()`  | 除了没有以`'\0'`结尾外，其他和`s.c_str()`同。                              |
| `s.copy()`  | 开辟新内存，返回指针。                                                  |

标准的string库就保证了用户能够显示的使用字符指针，而不轻易的造成错误。

#### 继承与构造函数

首先需要清楚编译器在创建一个对象（实例化）时做了什么。

1. 为对象开辟了一个内存。
2. 根据传来的形参类型和数量（也可为无）来选择构造函数来初始化。
3. 如果有的话，执行构造函数的函数体。

那么如果是创建一个含有基类的对象（继承实例化）时会怎么做呢？答案与上述几乎类似。

1. 为对象开辟内存。
2. 根据传来的形参类型和数量（也可无）对**派生类**的构造函数进行选择。
   - 注意，不是以先派生类后基类的顺序进行创建对象，实际上编译器选择派生类的构造函数来确定是否派生类的构造函数有对基类的数据显式的初始化，若没有显式，则基类执行隐式的默认初始化（基类中没有参数的构造函数）。
   - 然后再来先初始化基类，再初始化派生类。
3. 如果有的话，执行构造函数的函数体。

所以大部分情况下，派生类的初始化是先对基类进行默认无参数的初始化（隐式），然后在对派生类初始化（显式）。

#### 多态与虚拟函数

**多态**的含义就是程序能自动识别类型从而调用该类型对应的函数，即一种类型（基类）可以用作其他类型（基类或若干派生类）使用。

C++对于多态的支持是在类的**继承**的基础使用**虚拟函数**来实现的。

注意虚拟函数的使用是在基类的对应方法声明的前面加上**virtual**关键字，在调用方法的时候使用该类所属的基类类型的指针或者引用，编译器即可在运行的时候根据指针或引用的具体类型来调用对应的方法（需要在其他派生类中**重定义**该方法，且重定义时不需要`virtual`关键字）。

##### 虚拟析构函数

在使用`delete`函数时，编译器会先执行对应对象的析构函数，然后将内存返回系统。但是如果不适用虚拟特性的话，`delete`无法判断要使用基类析构还是派生类析构，这是使用虚函数即可让`delete`自动辨别使用哪个类的虚构函数。

##### 抽象基类和纯虚拟函数

通常纯在一种基类，他仅仅只是为了派生类而存在的，从来都不会对此类进行实例化，在其内拥有着只声明的虚拟函数，此为抽象基类，那个虚拟函数叫做纯虚拟函数，纯虚拟函数可能对当前基类没有实际意义，但是对各种派生类都有意义。

```c++
class Point {
public:
    virtual show_xy() const = 0;    // 若没有0只是会提醒你未定义，但有0则会直接报错：抽象基类不可实例化。
private:
    double x, y;
}
```

#### 常见重载运算符操作函数归纳

| 运算符                        | 实现思路                                                     |
| -------------------------- | -------------------------------------------------------- |
| `T &operator=(const T &)`  | 赋值的基本流程是先判断是否自我赋值。<br />若不为自我赋值，则释放原先内存，再新开辟内存以及新建立对象返回。 |
| `T &operator+=(const T &)` | 若仅仅是数字的`+=`那就相当于重新赋值吧。                                   |
| `T operator+(const T &)`   | 按值返回，赋值反正也要重新删除再开辟内存。                                    |

#### C++源文件(`.cpp`)是怎么被转换到可执行文件(`.exe`)的

大体上是从经过**编译**和**链接**两个过程实现的：

##### 编译

编译是将C++源文件转换为二进制的机器可识别代码文件的过程，获得中间文件`.obj`。

先进行**预处理**：将#后面的指令全部用其他代码代替，如`#define`将对下文所有代码对于的字符进行对应的替换，`#include`将其包含的文件的代码内容复制替换到源文件中，最后得到中间文件`.i`。

接下来就是一行一行的进行编译了。

##### 链接

链接是将上述所有`.obj`文件合在一起转换成可执行文件的过程。

对于每一个单独的C++源文件，所用到的函数必须声明或提前定义。

**只被声明的函数若被调用则必须在外面有定义，链接的目的就是找到那个定义，也就是在外部寻找声明的定义**

只被定义的函数也会被编译，这是因为有可能外面有其他文件需要用到这个函数定义（其他源文件链接到这个文件），故就算这个文件没有使用这个函数也会编译这个函数的定义。

只被定义的函数若添加关键词`static`，则就告诉编译器这个函数只在本文件中使用，故不存在其他文件链接到这里，故若这个函数没有用到这个函数，这段定义就成了死代码（没有作用）了。

##### 一种很常见的重复定义问题

你在头文件中定义，虽然你只在头文件中定义，但是在其他多个文件预处理时多次复制到其他文件，这样就形成了重复定义的问题。

解决方法有若干，在头文件中定义的那个函数添加`static`关键词表示在复制进入的文件中的该函数定义只能在那个函数使用。

其次就是添加`inline`关键词，调用代码处变成了内敛子程序（即将定义的代码嵌套到调用函数中），从此不再是调用，而是按顺序执行代码。

那么有没有一种可能不要再头文件中定义函数就可以解决这个问题呢？

#### 一些需要注意的地方

##### 关于自定义类的比较大小问题

若没有故意的定义，自定义类之间的比较大小是在逻辑运算符两边同时转换为bool类型（如果有定义转换类型函数），从而会导致错误，所以自定义类中比较是否相等（自我赋值）请**一定一定使用地址来判断`this != &rhs`**。（我特么找了一个晚上的bug才发现这个问题）

##### 指针与常量指针不能相互转换

比如你的某个函数以`const T *t`为其中一个参数，你在调用的时候传来的必须是常量指针，而不能是普通的指针。

#### 模板函数和模板特化函数

众所周知，模板函数可以涵盖所有类型的相似操作（即除了类型不同其他操作都相同的函数）。

为了实现某一特定功能，如果大部分类型对应该功能的操作是一样的（即可以用模板函数），但是偏偏就有那么一两种类型不能套用那个模板，该怎么办呢？答案就是模板特化，另外定义一个函数，编译器就会根据那个特殊的函数调用你定义的特化函数，格式如下。

```c++
class Ptr {
public:
    void make_unqiue() {
        if (*refptr != 1) {
            --*refptr;
            // 这一步若切换成 p = p ? p->clone : 0 就会出大问题，原因是不是所有类型都有clone()成员函数
            p = p ? clone(p) : 0;    
        }
    }
    ...
}

template<class T>
T *clone(const T *tp) {
    return tp->clone();
}
// m
template<>
Vec<char> *clone(Vec<char> *vp) {
    return new Vec<char>(*vp)
}
```

#### 变量的链接

毫无疑问，函数是一定可以链接的，这通常发生在我们将函数定义在其他文件，当在某个另外的文件要使用这个函数时只要事先声明，编译器就会在链接阶段将该文件的函数调用与定义函数的文件中对应的函数绑定（链接），从而达到层次清晰的目的。可以知道，发生链接的函数需要事先声明，并且定义需要在全局域（不知道这个名词准不准确）中。

实际上，变量也可以进行链接，通常发生在某个文件中要用到其他文件定义的全局变量。类似的，变量的链接也需要在使用该变量的文件中事先声明，并且必须是全局变量。而变量的声明就是使用`extern`关键字。

```c++
/* test1.cpp */
int s_Variable = 5;
/* test2.cpp */
extern s_Variable;		// 仅仅只对s_Variable这个变量进行声明，表示当前源文件要用到这个变量。
// 接下来就可以在test2.cpp文件中使用全局变量s_Variable了
```

#### static的作用

- 不管是对函数还是变量（都是全局，类外），使用此关键字可以保证该变量只能在文件内部使用，无法被外部文件链接。
- 在类中使用`static`，表示该变量或函数与类绑定而不是与类对象（实例）绑定，并且不能访问具体某个类对象（实例）的参数，必要时必须传入类类型，就像正常的定义全局函数一样。通常用于实例之间共享数据，为某个全局变量添加上特定的命名空间。

#### 左值和右值

左值就是占有内存中一定空间的值，比如某一个变量，在Accelerated C++中定义为非临时对象。右值则正好相反。

此概念在函数传参运用较为广泛，例如：

```c++
void lfunction(int &lvalue);        // lvalue即为左值引用，调用函数时必须传入左值
void rfunction(int &&rvalue);        // rvalue即为右值引用，调用函数时必须传入右值（临时变量）
void function(const int &value);    // 既可以传入左值，也可以传入右值，传入右值时编译器默认新建一个为左值的临时变量
```

#### 函数指针(pointer to function)简述

函数指针的类型由其对应返回类型和形参决定。

每个定义的函数的函数名都可以当作是一个常量指针，即当我们在调用函数的时候，写出的函数名会自动转换成指针。

函数指针声明方式：

```c++
bool (*pf)(const string &, const string &);
```

函数指针和函数类型一般情况下可以混用，但是唯独在函数返回类型的时候必须使用指针。

#### 文件的简单读写

包含文件的输入输出流的库是`<fstream>`。

基本用法：

```c++
#include <fstream>
int main {
    // 读取
    ifstream file_open("input.txt");
    // 输出
    oftream file_out("out.txt");
    // 随即就可对file_open和file_out进行各种与流有关的类似的操作。
    ...
    return 0;
}
```

规划一下后续的学习路线：

复习accelerated c++中的小结知识，结合C++ primer做好笔记。

大致浏览essential c++的内容，当作睡前读物。

做b站上的那个项目。

预期以上任务在期中后两周之前完成。

之后学习effective c++。

根据网课学习hands on design patterns with c++这本书。



**笑死，根本学不会模电，电磁场，你学你m的C++。**

**STM32怎么这么难学啊，硬件怎么这么难学啊，单片机怎么这么难学啊，焯！！**



#### 变量声明和定义的关系

为了支持分离式编译（多文件模块化编程），C++将声明和定义区分开来。**声明（declaration）**使得该名字为程序所知。**定义（definition）**使得名字与某个实体相关联。

一般情况下都是声明和定义一起实现的，例如

```c++
int i;		// 声明并定义了j
```

但是在C++中，变量只能被定义一次，而可以被声明多次，如果想在某一.c文件中使用另外一个.c文件定义的全局变量，则必须在该文件中先声明，让程序所知。例如

```c
extern int i;	// 仅仅只是声明了i，并没有对其定义
```

对于extern的理解不应该理解为外部变量意思，而应该理解其为单单声明变量的关键字，结合声明和定义的区别，即可清楚其用法。

注：当使用了extern关键字的时候，一定不能初始化，基于变量的初始化抵消了extern的作用。此时语句就变成了定义。

```c
extern double pi = 3.1416;	// 定义
```

#### 向上取整和向下取整

C/C++语言默认的整数除法是向下取整的，如
```c++
int x_1 = 1 / 8;	// 会得到0
int x_2 = 15 / 8;	// 会得到1
```

如果要向上取整该怎么办呢？可以使用以下方法
```c++
int x_3 = (1 + 7) / 8;
int x_4 = (15 + 7) / 8;
```

通过向分子加上除数减一，可以使的原本的数为1（向上取整的最小整数）时就可以发生进位，而在原本为0的地方不发生进位，即
```c++
floordiv(x, y) = x / y;		// floor:：地板，向下取整
ceildiv(x, y) = (x + y - 1) / y;	// ceil：天花板，向上取整
```

#### `[[nodiscard]]`标识符

该标识符添加到函数的前面，表示该函数的返回值是不能丢弃的，即必须使用
```c++
[[nodiscard]] int square_my(int x) {
    return x * x;
}

void fun() {
    int x = 1;
    square_my(x);			// 此时编译器会警告
    int y = square_my(x);	// 编译器不会警告
}
```

