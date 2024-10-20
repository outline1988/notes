## 双指针

#### 26. 删除有序数组中的重复项

给你一个 **升序排列** 的数组 `nums` ，请你原地 删除重复出现的元素，使每个元素 **只出现一次** ，返回删除后数组的新长度。元素的 **相对顺序** 应该保持 **一致** 。

由于在某些语言中不能改变数组的长度，所以必须将结果放在数组`nums`的第一部分。更规范地说，如果在删除重复项之后有 `k` 个元素，那么 `nums` 的前 `k` 个元素应该保存最终结果。

将最终结果插入 `nums` 的前 `k` 个位置后返回 `k` 。

不要使用额外的空间，你必须在 **原地修改输入数组** 并在使用 O(1) 额外空间的条件下完成。

**erase()**方法

```c++
iterator erase( iterator pos );
iterator erase( const_iterator first, const_iterator last );
```

删除迭代器`pos`位置的元素或删除迭代器`[first, last)`范围内的元素，返回删除的最后一个元素的下一个迭代器，即`pos + 1`或者`last`。

一定要注意，由于`vector`是数组结构，所以删除某些元素，会导致其后面的所有迭代器都失效，所以删除完后迭代器`pos`不再有效，老的`end()`也不再有效，所以通常用`erase()`的返回值来更新迭代器。

**循环嵌套**
首先用迭代器变量`iter`遍历数组，遇到重复出现的数字时开启第二层嵌套，即用迭代器`iter_inner`遍历完重复的元素，最后使用`num.erase()`方法将多余的重复元素删除。

```c++
int removeDuplicates(vector<int> &nums) {
    for (auto iter = nums.begin(); iter + 1 < nums.end(); ++iter) {
        auto iter_inner = iter + 1;
        if (*iter == *(iter + 1)) {		// 由于此判断不会造成终止循环，所以不能合并至循环的终止条件。
            while (iter_inner + 1 != nums.end() && *iter_inner == *(iter_inner + 1)) {
                ++iter_inner;
        }
        nums.erase(iter + 1, iter_inner + 1);
        }
    }
    return (int)nums.size();
}
```

需要注意的是，由于遍历的时候要将`iter`和`iter + 1`进行比较，所以若是正常的结束条件即`iter != nums.end()`，很有可能会超出数组边界，所以需要使用条件`iter + 1 < nums.end()`。

还需要注意的是，不要觉得使用`iter + 1 != num.end()`的条件就可以了，因为`iter`仍然有可能为`nums.end()`，由此这个终止条件便无法工作，所以只要遍历的时候要访问当前迭代器的后面，*一定要修改终止条件至包含有可能出现的迭代器超出边界的情况*。

还有就是关于循环内判断条件合并至循环终止条件的问题，必须得保证这个循环内的判断条件要结束循环才行。

**双指针1（一个元素一个元素删除）**
使用相邻的前后指针`it_lefr`和`it_right`，其中`it_right = it_left + 1`，对前后指针所指向的值进行判断，若相同，则删除`it_right`并重另`it_right = it_left + 1`，若不同两指针同时递增。

```c++
int removeDuplicates(vector<int> &nums) {
    auto it_left = nums.begin();
    auto it_right = it_left + 1;
    while (it_left != nums.end() && it_right != nums.end()) {
        if (*it_left == *it_right) {
            nums.erase(it_right);
            it_right = it_left + 1;
        }
        if (*it_left != *it_right) {
            ++it_left; 
            ++it_right;
        }
    }
    return (int)nums.size();
}
```

但是由于其删除元素是一个一个删除，所以最终会导致时间消耗很慢。

**双指针2（找到重复的头尾指针删除）**
在双指针1的思路上，遇见相同的元素不是直接删除，而是继续寻找相同元素的尾指针，从而同时删除一组元素。

```c++
int removeDuplicates(vector<int> &nums) {
    auto it_left = nums.begin();
    auto it_right = it_left + 1;
    while (it_left + 1 < nums.end() && it_right != nums.end()) {
        if (*it_left != *it_right) {
            ++it_left;
            ++it_right;
        }
        else if (*it_left == *it_right) {
            ++it_right;
        }
        if (it_right == nums.end() || *it_left != *it_right) {
            nums.erase(it_left + 1, it_right);
            it_right = it_left + 1;
        }
    }
    return (int)nums.size();
}
```

同一次循环，有且只能进行两个操作，一是递增操作，二是删除操作，二者不能相互交叉。*指针递增后一定要判断是否超出数组*，所以删除操作需要判断是否到达边界，由此再决定是否访问迭代器。

**使用C++泛型算法**

```c++
template< class ForwardIt >
ForwardIt unique( ForwardIt first, ForwardIt last );
```

`unique()`函数对传入[first, last)范围内的向量，进行删除相邻项，只保留一次重复的相邻量的操作，而被删除的重复的量被移至向量的末位，返回一个去重之后向量的最后一个元素的下一个迭代器，即重复元素形成子向量的第一个元素。

```c++
int removeDuplicates(vector<int> &nums) {
    auto it = unique(nums.begin(), nums.end());
    nums.erase(it, nums.end());
    return (int)nums.size();
}
```

关于更多的`unique()`函数的介绍请上cppreference。

#### 27. 移除元素

给你一个数组 `nums` 和一个值 `val`，你需要 原地 移除所有数值等于 `val` 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 `O(1)` 额外空间并 **原地修改输入数组**。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

**遍历法**

```c++
int removeElement(vector<int>& nums, int val) {
    for (auto iter = nums.begin(); iter != nums.end(); ++iter) {
        if (*iter == val) {
            iter = nums.erase(iter) - 1;    // 返回当前元素的下一个元素
        }
    }
    return nums.size();
}
```

**双指针法**

因为只要删除一次，所以直接用两次循环。

```c++
int removeElement(vector<int> &nums, int val) {
    sort(nums.begin(), nums.end());
    // 双指针
    auto it_left = nums.begin();
    auto it_right = it_left;
    while (it_left != nums.end() && *it_left != val) {
        ++it_left;
    }
    it_right = it_left;
    while (it_right != nums.end() && *it_right == val) {
        ++it_right;
    }
    nums.erase(it_left, it_right);
    return nums.size();
}
```

**双指针法2**
套模板，一次循环有且仅能进行递增和删除操作，关键在于删除操作的判断条件应该如何，比如此题应该包含`it_right`到达数组结尾的情况和实际应该进入删除操作的情况（要与递增的条件的相区分）。

```c++
int removeElement(vector<int> &nums, int val) {
    sort(nums.begin(), nums.end());
    // 双指针
    auto it_left = nums.begin();
    auto it_right = it_left;
    while (it_left != nums.end() && it_right != nums.end()) {
        if (*it_left != val) {
            ++it_left;
            ++it_right;
        }
        else if (*it_right == val) {
            ++it_right;
        }
        if (it_right == nums.end() || (*it_left == val && *it_right != val)) {
            nums.erase(it_left, it_right);
            break;
        }
    }
    return (int)nums.size();
}
```

#### 11. 盛最多水的容器

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

**双指针法减小搜索空间**
要使盛水区域最大，则要求底和高都尽量大，我们可以从拥有最大底边时（此时底边长为`n`）开始，我们无法知道此时的面积是否就是最大的面积，所以我们还得想办法看是否能找到更大的面积。

我们已经知道了拥有最大底边时的面积是多少，若想知道是否有其他面积比它更大，只能朝着底边减小而高度增大的方向进行，观察高度的增大是否能够弥补底边减小而带来的面积损失。由于高度取决于两边高度的最小值，所以让高度较小的那一边朝着中心前进，才能在遇到更高边的是否有可能让面积增加。
```c++
int maxArea(vector<int>& height) {
    auto left = height.begin();
    auto right = height.end() - 1;
    int max_area = 0;
    int temp_area;
    while (left != right) {
        temp_area = (int)(right - left) * min(*left, *right);
        max_area = max_area > temp_area ? max_area : temp_area;
        if (*left > *right) {	// 让高度较小的边朝着中心前进
            --right;
        }
        else {
            ++left;
        }
    }
    return max_area;
}
```

为什么这样做是正确的？
从底边最长的情况开始，此时面积要么就是最大，要么就是内部还有比这个面积更大的情况，假设此时的左边界高度小于右边界高度，即`*left < *right`，则要想寻找面积更大的情况，必须朝着底边减小而高增大的方向进行。假设内部存在最大面积的情况，那么此情况的`*left`一定比底边最长情况的`*left`要长，因为底边最长情况高度的瓶颈在于`*left`，所以一定要更高的`*left`才能有可能拥有更大的面积，所以要朝着更低高度向内的方向前进。

#### 167. 两数之和 II - 输入有序数组

给你一个下标从 **1** 开始的整数数组 `numbers` ，该数组已按 **非递减顺序排列** ，请你从数组中找出满足相加之和等于目标数 `target` 的两个数。如果设这两个数分别是 `numbers[index1]` 和 `numbers[index2]` ，则 `1 <= index1 < index2 <= numbers.length` 。

以长度为 2 的整数数组 `[index1, index2]` 的形式返回这两个整数的下标 `index1` 和 `index2`。

你可以假设每个输入 **只对应唯一的答案** ，而且你 **不可以** 重复使用相同的元素。

**双指针减小搜索空间**
由于是有序数组，所以可以让指针`left`和`right`分别从左右两边开始，朝着能够满足题目要求的条件往中间靠近。

```c++
vector<int> twoSum(vector<int>& numbers, int target) {
    auto iter_left = numbers.begin();
    auto iter_right = numbers.end() - 1;
    while (iter_left != iter_right && *iter_left + *iter_right != target) {
        if (*iter_left + *iter_right > target) {
            --iter_right;
        }
        else if (*iter_left + *iter_right < target) {
            ++iter_left;
        }
    }
    return {(int)(iter_left - numbers.begin()) + 1, (int)(iter_right - numbers.begin()) + 1};
}
```

#### 240. 搜索二维矩阵 II

难度中等1243收藏分享切换为英文接收动态反馈

编写一个高效的算法来搜索 `m x n` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

 **减小搜索空间**
锁定矩阵右上角的元素，该元素可增（向下移动）可减（向左移动），所以只需让其不断地向目标值逼近就可以了。

```c++
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int i = 0;
    int j = (int)matrix[0].size() - 1;
    while (i != (int)matrix.size() && j != -1) {
        if (matrix[i][j] > target) {
            --j;
        }
        else if (matrix[i][j] < target) {
            ++i;
        }
        else {
            return true;
        }
    }
    return false;
}
```

总结一下减小空间算法，其都是在于找到一个”中间值“，”中间“的含义并不是指元素值在中间，而是可以方便的向着目标值进行两种操作，若元素值比目标值大则进行从大往小方向的操作，若元素值比目标值小则进行从小往大的方向操作，中间值选取的关键就是能够方便的进行这**两项操作**而**不会遗漏其他元素**。

#### 3. 无重复字符的最长子串

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

**双指针法**
使用双指针`left`和`right`，让`right`每次循环都递增，而`left`确定每次查找子串的开头，每次循环判断`*right`是否包含在`[left, right)`范围内，若包含，则另`left`为找到元素的下一个元素。

```c++
int lengthOfLongestSubstring(string s) {
    auto left = s.begin();
    auto right = s.begin();
    decltype(left) temp;
    int max_substr_len = 0;
    while (left != s.end() && right != s.end()) {	// if left == right, cannot find, return right
        temp = find(left, right, *right);
        if (temp != right) {    // find it
            max_substr_len = (int)(right - left) > max_substr_len ? (int)(right - left) : max_substr_len;
            left = temp + 1;    // skip repeated element
        }
        ++right;
    }
    max_substr_len = (int)(right - left) > max_substr_len ? (int)(right - left) : max_substr_len;
    return max_substr_len;
}
```

判断`left`和`right`的初始值是否都为`s.begin()`，就把这样的条件带入循环，发现相同时`find()`相当于没发现元素，即返回`right`，由此相同时进行`++right`，所以可以相同。

**滑动窗口**
更方便的使用哈希表来查找。

## 二分查找

#### 35. 搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

**朴素算法（遍历数组）**

```c++
int searchInsert(vector<int>& nums, int target) {
    auto iter = nums.begin();
    while (iter != nums.end() && *iter < target) {
        ++iter;
    }
    return (int)(iter - nums.begin());
}
```

没啥好说的。

**二分查找**
任何的二分查找都可归结为初始值为`left = -1`和`right = nums.size()`的寻找边界（即目标值`k`或`k + 1`中间边界）的问题，而最终返回的下标就是该边界（由题目条件而定）的左边（即`k`）或右边（即`k + 1`），所以出循环的条件为`left + 1 == right`。

若将边界左边视为蓝色领域（范围为`[-1, k]`），边界右边视为红色领域（范围`[k + 1, nums.size()]`），则二分查找的过程就可视为两边领域从初始区间（蓝色`[-1]`和红色`[nums.size()]`）通过中值向边界快速扩张的过程。例如在第一次循环中，若中值属于蓝色领域（左边），则蓝色领域迅速从`[-1]`扩张至`[-1, mid]`，接下来的循环重复上述操作直到`left`和`right`卡出了边，即`left + 1 == right`，此时退出循环，返回`left`或`right`即可。

验证上述算法的可行性

1. 是否会陷入死循环（即是否能够成功跳出循环）
   查找至最终几步时，一定会有这几种情况`left + 2 == right`或`left + 3 == right`，前者的下一次循环`mid = left + (right - left) / 2 = left + 1 `，此时正好可以跳出循环，同理可证后者的情况也能满足。
2. 初值`left = -1`和`right = nums.size()`是否合理
   上述的模型是基于蓝红领域扩张的假设下而进行的，所以必然需要初始的蓝红领域，而`left = -1`和`right = nums.size()`时所产生的初始蓝红领域正好能够在后续中包含所有可能的蓝红领域的情况。若选择`left = 0`，则此时无法处理当数组中的所有元素都是红色领域的情况，因为选择`left = 0`就已经默认包含了其必须为蓝色领域，而不能包含后续的所有可能的蓝红分组情况，其他的同理。
3. `mid`是否会溢出（保证在循环内`mid`总能访问合法的数组元素）
   `mid = (left + right) / 2`，`mid`最小的情况为前一次循环的记过`left = -1`和`right = 1`（由此才能再次进入循环求出新一轮的`mid`），而此时最小的`mid = 0`，包含至边界内；同理，`mid`最大的情况出现在`left = nums.size() - 2`和`right = nums.size()`的情况下，此时最大的`mid = nums.size() - 1`，同样不会超出边界。
4. 选择返回`left`还是`right`
   若选择返回`left`，那么其范围为`[-1, length - 1]`，则往往会在`-1`处导致数组溢出；
   若选择返回`right`，那么其范围为`[0, length]`，其往往会在`length`处导致数组溢出。

具体可看视频[二分查找模型](https://www.bilibili.com/video/BV1d54y1q7k7)。

如本题中，可以通过题意判断出我们所要查找的就是升序数组中第一个出现的大于等于`target`的下标，所以可以分出边界为大于等于`target`和其补区间，由此写出
```c++
int searchInsert(vector<int> &nums, int target) {
    int left = -1;
    int right = (int)nums.size();
    int middle;
    while (left + 1 != right) {
        middle = left + (right - left) / 2;
        if (nums[middle] >= target) {
            right = middle;
        }
        else {
            left = middle;
        }
    }
    return right;
}
```

如果希望通过迭代器来实现二分查找，但是迭代器的`iter_left`仅能为`nums.begin()`，相当于`left = 0`，而不满足初始的边界条件，可以通过让`left`和`iter_left - 1`有相对应的关系，即先将全部的`left`用`left - 1`来替代，再将下标转化为迭代器即可，如下
```c++
int searchInsert(vector<int> &nums, int target) {
    auto left = nums.begin();
    auto right = nums.end();
    decltype(left) middle;
    while (left != right) {		// (left - 1) + 1 != right
        middle = left + (right - left - 1) / 2;	// middle = left - 1 + (right - left + 1) / 2
        if (*middle >= target) {
            right = middle;
        }
        else {
            left = middle + 1;	// left - 1 = middle
        }
    }
    return (int)(right - nums.begin());
    // return (int)(left - 1 - nums.begin());
}
```

#### 34. 在排序数组中查找元素的第一个和最后一个位置

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

**两次二分查找**
两种边界，按照上述使用两次双指针即可

```c++
vector<int> searchRange(vector<int>& nums, int target) {
    /*	可选择的特殊情况处理1
	if (find(nums.begin(), nums.end(), target) == nums.end()) {
            return {-1, -1};
    }
    */
    int left1, left2;
    int right1, right2;
    int mid1, mid2;
    left1 = left2 = -1;
    right1 = right2 = (int)nums.size();
    while (left1 + 1 != right1) {
        mid1 = left1 + (right1 - left1) / 2;
        if (nums[mid1] >= target) {
            right1 = mid1;
        }
        else {
            left1 = mid1;
        }
    }

    while (left2 + 1 != right2) {
        mid2 = left2 + (right2 - left2) / 2;
        if (nums[mid2] <= target) {
            left2 = mid2;
        }
        else {
            right2 = mid2;
        }
    }
    // 特殊情况处理2
    if (nums.size() == 0 || left2 == -1 || right1 == nums.size() || nums[left2] != target) {
        right1 = -1; left2 = -1;
    }
    return {right1, left2}; 
}
```

相比于特殊情况处理2，特殊情况处理1需要遍历一次数组，而处理2只需做几个判断即可，所以选择处理2。

**双指针法（效率较低）**
按照前面总结的双指针法模板写就好，如下

```c++
vector<int> searchRange(vector<int> &nums, int target) {
    auto iter_left = nums.begin();
    auto iter_right = nums.begin()
    while (iter_left != nums.end() && iter_right != nums.end()) {
        if (*iter_left != target) {
            ++iter_left;
            ++iter_right;
        }
        else if (*iter_right == target) {
            ++iter_right;
        }
        if (iter_right == nums.end() || (*iter_left == target && *iter_right != target)) {
            break;
        }
    }
    if (iter_left == nums.end()) {
        return {-1, -1};
    }
    return {(int)(iter_left - nums.begin()), (int)(iter_right - 1 - nums.begin())};
}
```

理论上来说，这个双指针的时间复杂度要比二分查找的时间复杂度要高，但是leetcode上运行的时间要更短。。。

此题解法同样适用与“27. 移除元素”。

#### 69. x 的平方根

给你一个非负整数 `x` ，计算并返回 `x` 的 **算术平方根** 。

由于返回类型是整数，结果只保留 **整数部分** ，小数部分将被 **舍去 。**

**二分查找（连续）**
仍然按照上述二分查找的模型，只不过之前是离散的，而这次是连续的，然而连续的初值条件直接就是0和结尾，编写代码要方便得多。由之前的思路，需要找到蓝红区域，从而卡出边界，可以很轻松想到红色区域就是`mid * mid > target`，那么蓝色区域也随之确定。理想状态下边界就是实际的`sqrt(x)`的值，即理想的最终跳出循环边界为`left = right`，然而误差是无法避免的，所以最后的边界为`right - left <= error`。

```c++
double mySqrt(double x) {
    double x_temp = x < 1 ? 1 / x : x;

    double left = 0;
    double right = x_temp;
    double mid;
    double error = 1e-3;
    while (right - left > error) {
        mid = (left + right) / 2;
        if (mid * mid > x_temp) {
            right = mid;
        }
        else {
            left = mid;
        }
    }
    return x < 1 ? 1 / right : right;
}
```

需要注意的是`x`小于1是需要双重倒数处理。

**二分查找（离散）**
由于题目要求函数的参数和返回值都是整数，所以可不不比使用浮点数的计算方法，转而使用离散的二分查找（数组中使用二分查找），最终的边界为`left + 1 == right`，而可确定红色领域为`mid * mid > x`，最终返回的是边界左边的`left`（由于返回的值只保留整数部分，所以要么与真值相同，要么小于真值）。

```c++
int mySqrt_int(int x) {
    int index_left = -1;
    int index_right = x + 1;
    int index_mid;
    while (index_left + 1 != index_right) {
        index_mid = index_left + (index_right - index_left) / 2;
        if (index_mid * index_mid > x) {
            index_right = index_mid;
        }
        else {
            index_left = index_mid;
        }
    }
    return index_left;
}
```

**牛顿迭代法**
牛顿迭代法可以用来求解非线性方程的根，其核心在于不断地通过近似线性的方法来逼近真实值，具体推导不谈，只有以下公式
$$
x_{k + 1} = x_k - \frac{f(x_k)}{f^{\prime}(x_k)}
$$
只要选择合适的初值$x_0$，则可不断的按照上式迭代，最终收敛至真实值。

对于求平方根的情况，可另$f(x) = x^{2} - a$，然后根据上式可列为
$$
x_{k+1} = \frac{1}{2}(x_k + \frac{a}{x_k})
$$
由对勾函数的性质可知道，当初值$x_0 > \sqrt{a}$或$x_0 < \sqrt{a}$时，$x_k$都会不断地向$\sqrt{a}$的方向去逼近，所以最终代码如下

```c++
double mySqrt_Newton(double x) {
    double x_0 = x + 0.1;	// 为了让x = 0时也可以进入迭代
    double x_k, x_k_1;
    double error = 1e-5;
    x_k_1 = x_0;
    do {
        x_k = x_k_1;
        x_k_1 = (x_k + x / x_k) / 2;
    } while (fabs(x_k_1 - x_k) > error);
    return x_k_1;
}
```

跳出循环条件选择为`fabs(x_k_1 - x_k) <= error`，这是由于`x_k`和`x_k_1`的距离会随着`x_k`不断逼近$\sqrt{a}$过程而减小，所以距离越小越逼近，故可使用此距离作为跳出循环条件。（还有很多的数学公式没有推导，要用到的时候再说吧）

#### 33. 搜索旋转排序数组（就是DSP循环移位）

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

**二分查找**
大致的思路就是，我们先将循环移位数组还原为升序数组，再对升序数组进行二分查找，所查找到的下标在进行循环移位即可。

若完全按照上述思路，需要的时间复杂度会很大，所幸二分查找中需要升序数组的信息唯有判断`middle`是属于蓝色区域还是红色区域，然而这个操作我们完全可以可以将升序数组的`middle`做个循环移位的映射至循环移位数组的`middle`值，由此得到的最终是升序数组的下标，故还需再次循环移位才能得到正确答案。
```c++
int circular_shift(int index, int shift_num, int length) {  // 右移shift_num后的新下标
    return (index + shift_num) % length;    // 1 <= shift_num <= length
}

int search(vector<int>& nums, int target) {
    auto iter = nums.begin();
    while (iter + 1 != nums.end() && *iter < *(iter + 1)) {
        ++iter;
    }
    int shift_num = ( (int)(iter - nums.begin()) + 1 ) % (int)nums.size();
    int length = (int)nums.size();


    int left = -1;
    int right = (int)nums.size();
    int middle;
    while (left + 1 != right) {
        middle = left + (right - left) / 2;
        if (nums[circular_shift(middle, shift_num, length)] > target) {
            right = middle;
        }
        else {
            left = middle;
        }
    }
    int ans = circular_shift(left, shift_num, length);

    return left == -1 || nums[ans] != target ? -1 : ans;    // 题目说nums不为空所以只需这些判断
}
```

#### 162. 寻找峰值

峰值元素是指其值严格大于左右相邻值的元素。

给你一个整数数组 `nums`，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 **任何一个峰值** 所在位置即可。

你可以假设 `nums[-1] = nums[n] = -∞` 。

**二分查找**
本题还是可以使用二分查找，先假设数组`nums`仅有一个峰值，那么我们可以将蓝色区域定义为*上升的区域*`nums[mid] < nums[mid + 1]`，而红色区域定义为*下降的区域*`nums[mid] < nums[mid + 1]`。

若有很多峰值，进行`mid`计算时会自动忽略一部分的峰值，从而缩小峰值的个数，最终只会压缩到一个峰值上，题目要求任意返回一个峰值，所以该算法可行。

```c++
int findPeakElement(vector<int>& nums) {
    int left = -1;
    int right = (int)nums.size();
    int mid;
    while (left + 1 != right) {
        mid = left + (right - left) / 2;
        if (mid + 1 == (int)nums.size() || nums[mid] > nums[mid + 1]) {
            right = mid;
        }
        else {  // 不会出现相等的情况
            left = mid;
        }
    }
    return right;
}
```

`mid + 1 == (int)nums.size()`的判断是为了防止`mid + 1`导致的数组溢出，同时其也有防止`right`溢出的作用。

选择`right`由此不会出现再`-1`处溢出，而至于`length`处，`mid + 1 == (int)nums.size()`时的判断可以使`right`永远不会为`length`，因为`mid == length - 1`是由`left == length - 2 || left = length - 3`和`right == length`造成的，所以此时让`right == mid`即`right = length - 1`就避免了其数组溢出。

二分查找一般返回值的意义为数组下标时，一般都是返回`right`。

#### 使用STL的二分查找

**`std::lower_bound && std::upper_bound `**

```c++
template< class ForwardIt, class T, class Compare >
constexpr ForwardIt lower_bound( ForwardIt first, ForwardIt last,
                                 const T& target, Compare comp );
template< class ForwardIt, class T, class Compare >
constexpr ForwardIt upper_bound( ForwardIt first, ForwardIt last,
                                 const T& target, Compare comp );
```

`lower_bound`和`upper_bound`都是stl中内置的用于二分查找的函数，沿用上述想法，即二分查找是拓展领域直至卡出边界的过程，则`lower_bound`的蓝色领域为`comp(element, target)`函数为`true`的范围，返回值为边界的右边；`upper_bound`的红色领域为`comp(target, element)`为`true`的范围，返回值为边界的左边，注意这两个函数中的`comp`的参数顺序是不同的。

默认的`comp`为`<`，则根据上述，`lower_bound`返回的是第一个`>= target`的元素，而`upper_bound`返回第一个`> target`的元素。

#### SAR ADC

虽然这个话题是属于硬件领域的，但是其非常巧妙的使用了二分的思想，所以顺便写在二分查找的专题。

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

*拷于2023.6.17的Mastering STM32 Notes2.md*

### 哈希查找

#### 1. 两数之和

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

**哈希查找**
要求找到数组上两个相加为`target`的元素并以向量返回，该思路为新建一个数组，对于每个在`nums`中遍历过的元素来说，若在新建数组能够找到于其对应的元素，那么直接返回，若不能找到则添加进入新建数组，供其他元素查找。

而哈希查找的意思就是使用哈希的思想来进行查找这一过程。
```c++
vector<int> twoSum(vector<int>& nums, int target) {
    vector<int> ans;
    map<int, int> hashtable;
    auto iter_begin = nums.begin();
    for (auto iter = nums.begin(); iter != nums.end(); ++iter) {
        auto temp = hashtable.find(target - *iter);
        if (temp != hashtable.end()) {  // 在哈希表中查找
            return {(int)(iter - iter_begin), 1};
        }
        hashtable[*iter] = (int)(iter - iter_begin);	// 存入哈希表
    }
}
```

****

### 杂项（我也不知道归到哪里）

#### 9. 回文数

给你一个整数 `x` ，如果 `x` 是一个回文整数，返回 `true` ；否则，返回 `false` 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

**整型转字符串**
这里主要要掌握的是C++中整型转化为字符串的操作方法，用的时候再来查

```c++
bool isPalindrome(int x) {
    // int -> string
    ostringstream os;
    os << x;
    string str_x = os.str();
    
    auto left = str_x.begin();
    auto right = str_x.end() - 1;
    while (left < right && *left == *right) {
        ++left;
        --right;
    }
    return *left == *right ? true : false;
}
```

上述代码需要包含`sstream`库。

**栈方法**
由于回文串的前半数字和后半数字是对称的，所以只要抛去奇数位数情况下中间的值，就可以通过相同栈元素（前半和后半对应的元素是否相同）匹配后弹出，最后判断栈是否为空的方式来进行。

```c++
bool isPalindrome(int x) {
    if (x < 0) {
        return false;
    }
    stack<int> x_stack;
    int remainder;
    int decimal_count = 1;
    int decimal_count_result;
    int x_copy = x;
    while (x > 0) {
        remainder = x % 10;
        decimal_count_result = x / decimal_count;
        if (decimal_count_result > 0 && decimal_count_result < 10) {
            decimal_count *= 10;    // 此时decimal_count_result是最高位的数
        }
        else if (decimal_count_result > 9) {
            x_stack.push(remainder);
            decimal_count *= 10;
        }
        else if (decimal_count_result == 0 && x_stack.top() == remainder) {
            x_stack.pop();
        }
        x /= 10;
    }
    return x_stack.empty();
}
```

代码写的很长很繁琐，繁琐的地方在于需要将数字的后半先加入栈后再来在前半判断是否弹出，并且还要抛去中间元素的影响。

这里使用`decimal_count`来判断是否进行过半，`decimal_count`美国一个后半的循环，就会乘10，如此过半之后，`x / decimal == 0`，此外其还可以判断某一位数是否为中间元素。

**反转所有位数**
将整个数字进行反转，若相同则回文数

```c++
bool isPalindrome(int x) {
    if (x < 0) {
        return false;
    }
    int x_copy = x;
    int ans = 0;	// 提交时要改为long ans = 0;
    while (x_copy > 0) {
        ans = ans * 10 + x_copy % 10;
        x_copy /= 10;
    }
    return ans == x;
}
```

由于回文串反转和原来相同，那么原来不会溢出就代表着反转后也不会溢出，所以是可以使用原来`int`来装下反转后的数字的，治于会发生溢出的情况，则一定能判断其不为回文串，正好也可以通过反转后的串和原来串相比来比较，但是在提交中系统会敏锐的察觉是否溢出从而报错。

**反转半个位数**
将数字的后半（包括中间值）反转，看起与前半是否相同。

```c++
bool isPalindrome(int x) {
    if (x < 0 || (x % 10 == 0 && x != 0)) {
        return false;
    }
    int ans = 0;
    while (x > ans) {
        ans = ans * 10 + x % 10;
        x /= 10;
    }
    return x == ans || x == ans / 10;
}
```

#### 排序索引

可以很方便的调用C++的模板函数`std::sort`来进行排序元素，但是一般的操作会改变原数组的相对位置，且得到的数组损失了原来的位置信息。可以通过对下标排序而使用原始数组的值来比较，从而实现下标数组的排序，如下
```c++
void Sigvector::fft_sort_index() {
    std::iota(fft_mag + len / 2, fft_mag + len, 0);
    std::sort(fft_mag + len / 2, fft_mag + len,
         [&](size_t index_1, size_t index_2) { return fft_mag[index_1] > fft_mag[index_2]; } );
}
```

这里，`fft_mag`为一个`len`长度的数组（原谅我在C++中使用内置数组），前`len / 2`为排序参考的元素，后`len / 2`被我们设定为下标数组，也就是我们实际要排序的数组。通过匿名函数来将下标数组中的元素（代表下标）传入排序参考数值中，并用匿名函数加以体现。实际上我通过参考数组的值来以相同的方式排序下标数组，改变下标数组的相对位置，从而使下标数组每个元素（下标）所对应的排序参考数组的元素是排完序的。

#### 寻找峰值算法

scipy和matlab都有对应的find peaks函数，那么C++中要怎么进行移植呢？

