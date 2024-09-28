## gdb 基础指令

- `info r`: 显示通用寄存器的值。
- `info stack`: 显示当前堆栈帧的信息。
- `b 74`: 在第74行设置断点。
- `run`: 运行程序。
- `disas`: 反汇编。
- `p`: 打印变量的值。
- `x`: 显示内存中的内容。
- `layout`：调试可视化神器，有点像dos中的指令界面。
- `stepi`：继续执行程序下一行源代码中的汇编指令。如果是函数调用，这个命令将进入函数的内部，单步执行函数中的汇编代码。
- `finsh`：跳出函数。 
- `delete n`：删除n号断点。
- `clear n`：清除行n上面的所有断点。 
- `display/3i $pc`：展示后续3行指令。

## Lab
### Phase_1
![[storage bag/Screenshot 2023-11-26 at 15.56.04.png]]
> 开头的`sub 8`和结尾的`add 8`就是比较典型的入栈、出栈过程。

函数`call`了一个`strings_not_eauql`，如推断出如果输入字符串不等，炸弹就会爆炸。

去查看`0x402400`的内容：

![](storage%20bag/Screenshot%202023-11-26%20at%2016.14.32.png)

输入答案，完成phase_1:

![](storage%20bag/Screenshot%202023-11-26%20at%2016.17.46.png)

### Phase_2
![[storage bag/Screenshot 2023-11-26 at 16.22.28.png]]
> `rbp`，`rbx`是被调用者保存寄存器

可以看出，程序做了一件“读6个数”的事。
首先，如果`0x1`和`*rsp`不等，则炸弹爆炸，可以推断出第一个数是1。

<+30>处对eax中的值做了x2操作，不等则炸弹爆炸，故第二个数是2。接下来的内容类似一个循环结构，故剩下的结果依次翻倍即可得到。

![](storage%20bag/Screenshot%202023-11-26%20at%2017.21.54.png)

### Phase_3

在89行打上断点。
![[storage bag/Screenshot 2023-11-26 at 17.34.39.png]]


![](storage%20bag/Screenshot%202023-11-26%20at%2017.35.14.png)
> 可以看出，输入参数应该是两个*Decimal*，去ans.txt里随便输俩数。

![[storage bag/Screenshot 2023-11-26 at 17.45.22.png]]
> +39行指令，如果大于的话跳到106(`0x8(%rsp) - 0x7`>0)，也就是炸弹爆炸。
![[storage bag/Screenshot 2023-11-26 at 17.50.20.png]]

![](storage%20bag/Screenshot%202023-11-26%20at%2017.52.14.png)
这就是在ans.txt里写的一个数。

因此将第一个数改为小于7的数后再调试，这里选择了5：
```asm
<+85>:    mov    $0xce,%eax
```
> [!tips]
> eax = 206

![](storage%20bag/Screenshot%202023-11-26%20at%2018.33.32.png)
> $rsp + 0xc = 206

![](storage%20bag/Screenshot%202023-11-26%20at%2018.34.12.png)
> 发现这里刚好是txt中的第二个数

因此，答案为`5:206`

第一个数会影响第二个数的答案，原因就是<+51>-<+104>行指令类似C语言中的`switch`。
![[storage bag/Screenshot 2023-11-26 at 18.58.02.png]]
> 另一组答案`4:389`

### Phase_4

```C
0x400fce <func4>        sub    $0x8,%rsp                                                        
0x400fd2 <func4+4>      mov    %edx,%eax    //将edx=14 放到eax中                                                   
0x400fd4 <func4+6>      sub    %esi,%eax    //将esi=0与eax=14相减                                                    
0x400fd6 <func4+8>      mov    %eax,%ecx    //ecx=14                                                    
0x400fd8 <func4+10>     shr    $0x1f,%ecx   //对ecx中的值（14）逻辑右移31位 ，ecx=0                                                   
0x400fdb <func4+13>     add    %ecx,%eax    //eax=14                                                    
0x400fdd <func4+15>     sar    %eax         //sar 算数右移1位，相当于除以2，eax=1110>>1=0x7                                                    
0x400fdf <func4+17>     lea    (%rax,%rsi,1),%ecx  // ecx = 1 * rsi + rax =1*0+0x7=0x7                                            
0x400fe2 <func4+20>     cmp    %edi,%ecx      //edi就是输入的第一个参数，ecx-edi=0x7-edi                                                  
0x400fe4 <func4+22>     jle    0x400ff2 <func4+36> //如果<=0（edi>=7），就跳转到func4+36                                            
0x400fe6 <func4+24>     lea    -0x1(%rcx),%edx     //如果edi<7就edx-1 ，edx = rcx-1                                           
0x400fe9 <func4+27>     callq  0x400fce <func4>    //进入func4                                            
0x400fee <func4+32>     add    %eax,%eax                                                        
0x400ff0 <func4+34>     jmp    0x401007 <func4+57>                                              
0x400ff2 <func4+36>     mov    $0x0,%eax   //eax=0                                                     
0x400ff7 <func4+41>     cmp    %edi,%ecx   //ecx-edi                                                     
0x400ff9 <func4+43>     jge    0x401007 <func4+57>  //如果>=0（edi<=7），就跳出函数                                            
0x400ffb <func4+45>     lea    0x1(%rcx),%esi       //否则，esi=rcx+1                                      1
0x400ffe <func4+48>     callq  0x400fce <func4>     //重新执行func4函数                                         
0x401003 <func4+53>     lea    0x1(%rax,%rax,1),%eax                                            
0x401007 <func4+57>     add    $0x8,%rsp                                                        
0x40100b <func4+61>     retq 
```

### Phase_5

![[storage bag/Screenshot 2023-12-07 at 15.45.50.png]]
可以发现这里是在用自己输入答案的字符串长度与6对比，不等则explode。

![[storage bag/Screenshot 2023-12-07 at 16.00.12.png]]

`0x40108b <phase_5+41>   movzbl (%rbx,%rax,1),%ecx`

![[storage bag/Screenshot 2023-12-07 at 15.53.12.png]]
> 发现rbx指向了我们输入的答案的最低位

![[storage bag/Screenshot 2023-12-07 at 15.57.35.png]]
> 这里cl是ecx的低位，所以内容一样。

'1' = 00110001，所以`$0xf,%edx`  就是取'1'的低四位：'0001'

在 32 位 CPU 中，rdx 和 edx 是同一个寄存器，位宽相同只有高低位的区别

查看0x4024b90，发现了隐藏的字符串

![[storage bag/Screenshot 2023-12-07 at 16.07.23.png]]

```text
"maduiersnfotvbylSo you think you can stop the bomb with ctrl-c, do you?"
```

![[storage bag/Screenshot 2023-12-07 at 16.11.23.png]]
> 循环

![[storage bag/Screenshot 2023-12-07 at 16.18.32.png]]

![[storage bag/Screenshot 2023-12-07 at 16.20.14.png]]

![[storage bag/Screenshot 2023-12-07 at 16.21.48.png]]

可以大概得出伪代码：
```C
string s = "maduiersnfotvbylSo you think you can stop the bomb with ctrl-c, do you?

for(int i = 0;i != 6; i++){
	phase_5 + 41

	char ch = arr[i]
		s[ch] -> rsp + 10 + i * sizeof(char)
	phase_5 + 74

}
```

ch1, ch2, ch3, ch4, ch5, ch6 & 0xf

![](storage%20bag/Screenshot%202024-01-28%20at%2018.00.09.png)

而要依靠索引来判断是否相等的答案'flyers'在字符串中的索引顺序：9,15,14,5,6,7
所以低四位应分别为：1001, 1111, 1110, 0101, 1010, 0111

前面又限制了字符串长度，故答案为字母。

```sh
A 0100 0001
Z 0101 1010

0100 1001 -> 73 -> I
0100 1111 -> 79 -> O
0100 1110 -> 78 -> N
0100 0101 -> 69 -> E
0100 1010 -> 70 -> F
0100 0111 -> 71 -> G
# 不唯一，如小写也行，低4位一致即可
```

### Phase_6

![](storage%20bag/Screenshot%202024-01-28%20at%2018.21.06.png)

- 第一个数字减1后不大于5，即小于等于6

![](storage%20bag/Screenshot%202024-01-28%20at%2018.44.20.png)

```cpp
for (int i = 0; i <= 6; i++){
	if (num[i] > 6)
		bomb
	for(int j = i + 1; j <= 6; j++){
		if(nums[i] == num[j])
			bomb
	}
}
```

在循环外的`0x401153`，此处功能为：
```cpp
a[i] = 7 - a[i]
```

```cpp
a[i] > a[i+1]
```

![](storage%20bag/Pasted%20image%2020240128185938.png)

$\therefore$ node : 3 -> 4 -> 5 -> 6 -> 1 -> 2

7 - node = 4, 3, 2, 1, 6, 5

