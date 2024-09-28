### howManyBits
```C
/* howManyBits - return the minimum number of bits required to represent x in
 *             two's complement
 *  Examples: howManyBits(12) = 5
 *            howManyBits(298) = 10
 *            howManyBits(-5) = 4
 *            howManyBits(0)  = 1
 *            howManyBits(-1) = 1
 *            howManyBits(0x80000000) = 32
 *  Legal ops: ! ~ & ^ | + << >>
 *  Max ops: 90
 *  Rating: 4
 */
```

考虑情况：
`-1`：形如n个1，只需1位表达（极端情况下`0x1`=-1）；
`0`：一位0即可表示；
`> 0`：1 + 符号位0
`> 0`：直接返回所有位

```C
int howManyBits(int x) {
	int flag = x >> 31; // < 0 : 11111111; > 0 : 00000000 
	x = ((~flag) & x) | (flag & (~x));
```

对于正数，x为原值；对于负数，截取除符号表示用途以外的最高位（如11111001前面的1只用1位即可表示）

计数模板：
高`x`位是否包含任意非零位，若为真则至少需要`x`位来表示，故`计数+=x`
```C
	int bit_16 = (!(!!(x >> 16)) ^ 0x1) << 4; // !!为归一化操作
	x >>= bit_16;
```

经过化简后：
```C
	int bit_16 = (!(x >> 16) ^ 0x1) << 4;
	x >>= bit_16;
	
	int bit_8 = (!(x >> 8) ^ 0x1) << 3;
	x >>= bit_8;
	
	int bit_4 = (!(x >> 4) ^ 0x1) << 2;
	x >>= bit_4;
	
	int bit_2 = (!(x >> 2) ^ 0x1) << 1;
	x >>= bit_2;
	
	int bit_1 = (!(x >> 1) ^ 0x1);
	x >>= bit_1;
	
	int bit_0 = x;
	
	return bit_16 + bit_8 + bit_4 + bit_2 + bit_1 + bit_0 + 1;  
```

### floatScale2

```C
//float
/* 
 * floatScale2 - Return bit-level equivalent of expression 2*f for
 *   floating point argument f.
 *   Both the argument and result are passed as unsigned int's, but
 *   they are to be interpreted as the bit-level representation of
 *   single-precision floating point values.
 *   When argument is NaN, return argument
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
```

先通过掩码获取 s、exp、frac。

```C
	unsigned sign, exp, frac;
	sign = uf >> 31 & 0x1;
	exp = uf >> 23 & 0xFF;
	frac = uf & 0x7FFFFF;
```

考虑 `NaN` `Infinity` `0` 这三种直接返回参数的情况：
```C
	/* if uf is zero 
	if(exp == 0 && frac == 0)
	return 0;
	
	// infinity or Nah
	if(exp == 0xFF)
	return uf;
	*/
	if((exp == 0 && frac == 0) || exp == 0xFF)
		return uf;
```

由于非规格化和规格化的浮点数的值[[../02 信息的存储#^4db7aa|解释方法不同]]，因此要分情况讨论：

非规格化：
```C
	// denormalized : in IEEE 754 , E = 1 - bias, bias = 2^(k-1)-1.
	if(exp == 0){
	// E = 1 - 127 = -126
	frac <<= 1; // *2
	return (sign << 31) | frac;
	}
```

规格化：
```C
	// normalized
	exp++; // *2
	return (sign << 31) | (exp << 23) | frac;
```

### floatFloat2Int
```C
/* 
 * floatFloat2Int - Return bit-level equivalent of expression (int) f
 *   for floating point argument f.
 *   Argument is passed as unsigned int, but
 *   it is to be interpreted as the bit-level representation of a
 *   single-precision floating point value.
 *   Anything out of range (including NaN and infinity) should return
 *   0x80000000u.
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
```

同样，先通过掩码获取 s、exp、frac。
```C
	unsigned exp, sign, frac;
	int E, V;
	sign = uf >> 31 & 0x1;
	exp = uf >> 23 & 0xFF;
	frac = uf & 0x7FFFFF;
```
分情况讨论：
1. `NaN` `Infinity`， 返回`0x80000000u`；
```C
	if(exp == 0xFF)
		return 0x80000000;
```
2. `0`，返回 0；
```C
	if(exp == 0)
		return 0;
```
3. 非规格化的数，$E = 1 - 127 = -126， M_{max}=1, \therefore V = 0$
```C
	// denormalized
	if (exp == 0)
		return 0; // 可与分类 1 合并
```
4. 规格化的数，需要讨论 E 的大小：
```C
	E = exp - 127;
	V = frac | 1 << 23; // 把 E 乘到了 frac 上
	
	if(E > 31) // 位数溢出
		return 1 << 31;
	
	if(E < 0) // 过小
		return 0;
	
	if(E >= 23) // 可以保留全部尾数，左移，多的补0
		V <<= E - 23;
	else // 2^E位数不足以保留全部尾数，右移，省去尾数的部分位
		V >>= 23 -E; 
	
	if(sign) // 负数返回正数
		return ~V + 1;
	
	return V;
```

### floatPower2
```C
/* 
 * floatPower2 - Return bit-level equivalent of the expression 2.0^x
 *   (2.0 raised to the power x) for any 32-bit integer x.
 *
 *   The unsigned value that is returned should have the identical bit
 *   representation as the single-precision floating-point number 2.0^x.
 *   If the result is too small to be represented as a denorm, return
 *   0. If too large, return +INF.
 * 
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. Also if, while 
 *   Max ops: 30 
 *   Rating: 4
 */
```
> 这道题如果机器性能差的话可能跑不完报错，去`btest.c`文件中修改`#define TIMEOUT_LIMIT 20`即可。
```C
// E = exp - bias, exp = E + bias
  int exp = x + 127;
  // 0
  if (exp <= 0)
    return 0;
  // INF
  if (exp >= 0xFF)
    return 0x7f800000;
  return exp << 23;
```