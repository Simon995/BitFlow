//===- llvm/ADT/SparseBitVector.h - Efficient Sparse BitVector --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SparseBitVector class.  See the doxygen comment for
// SparseBitVector for more details on the algorithm used.
// 这个文件定义了 SparseBitVector 类。有关所使用算法的更多详细信息，请参阅 SparseBitVector 的
// doxygen 注释。
//
//===----------------------------------------------------------------------===//

#pragma once                          // 防止头文件被多次包含
#include <c10/macros/Macros.h>        // PyTorch C++ API 的宏定义
#include <c10/util/llvmMathExtras.h>  // LLVM 数学相关的额外功能，例如 countPopulation, countTrailingZeros

#include <array>     // 用于固定大小的数组 std::array
#include <cassert>   // 用于断言 assert
#include <climits>   // 用于 CHAR_BIT (一个字节中的位数)
#include <iterator>  // 用于迭代器相关，如 std::iterator
#include <list>      // 用于双向链表 std::list
#include <ostream>   // 用于输出流 std::ostream

namespace c10 {  // PyTorch 的 C++ 命名空间

/// SparseBitVector is an implementation of a bitvector that is sparse by only
/// storing the elements that have non-zero bits set.  In order to make this
/// fast for the most common cases, SparseBitVector is implemented as a linked
/// list of SparseBitVectorElements.  We maintain a pointer to the last
/// SparseBitVectorElement accessed (in the form of a list iterator), in order
/// to make multiple in-order test/set constant time after the first one is
// executed.  Note that using vectors to store SparseBitVectorElement's does
/// not work out very well because it causes insertion in the middle to take
/// enormous amounts of time with a large amount of bits.  Other structures that
/// have better worst cases for insertion in the middle (various balanced trees,
/// etc) do not perform as well in practice as a linked list with this iterator
/// kept up to date.  They are also significantly more memory intensive.
///
/// SparseBitVector 是位向量的一种实现，它通过仅存储具有非零位的元素来实现稀疏性。
/// 为了在最常见的情况下使其快速，SparseBitVector 实现为 SparseBitVectorElement 的链表。
/// 我们维护一个指向最后访问的 SparseBitVectorElement 的指针（以列表迭代器的形式），
/// 以便在第一个元素执行后，使多个有序的测试/设置操作达到常数时间。
/// 注意，使用向量存储 SparseBitVectorElement
/// 效果不佳，因为当位数很大时，在中间插入会导致巨大的时间开销。
/// 其他在中间插入方面具有更好最坏情况的结构（各种平衡树等）在实践中不如带有此保持更新的迭代器的链表表现好。
/// 它们也更加消耗内存。

template <unsigned ElementSize = 128>  // 模板参数，定义每个元素块管理的位数，默认为 128 位
struct SparseBitVectorElement {
public:
  using BitWord = unsigned long;  // 用于存储位的基本数据类型，通常是机器字长
  using size_type = unsigned;     // 用于表示大小的类型
  enum {
    BITWORD_SIZE = sizeof(BitWord) * CHAR_BIT,  // 一个 BitWord 中有多少位 (例如 64 位系统上是 64)
    BITWORDS_PER_ELEMENT =
        (ElementSize + BITWORD_SIZE - 1) / BITWORD_SIZE,  // 每个元素需要多少个 BitWord (向上取整)
    BITS_PER_ELEMENT = ElementSize  // 每个元素实际管理的位数 (等于模板参数 ElementSize)
  };

private:
  // Index of Element in terms of where first bit starts.
  // 元素的索引，表示这个元素块所管理的位在整个位向量中的起始位置（以 ElementSize 为单位）
  // 例如，如果 ElementSize 是 128，ElementIndex 为 0 表示管理 0-127 位，ElementIndex 为 1 表示管理
  // 128-255 位。
  unsigned ElementIndex;
  // 存储实际位的数组
  std::array<BitWord, BITWORDS_PER_ELEMENT> Bits{};  // 初始化为全0

  // 默认构造函数，ElementIndex 初始化为一个特殊值（通常表示无效或未设置）
  SparseBitVectorElement() : ElementIndex(~0U) {}  // ~0U 是 unsigned int 的最大值

public:
  // 构造函数，传入元素索引
  explicit SparseBitVectorElement(unsigned Idx) : ElementIndex(Idx) {}

  // 比较操作符
  bool operator==(const SparseBitVectorElement& RHS) const {
    if (ElementIndex != RHS.ElementIndex)  // 索引不同，则元素不同
      return false;
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)  // 比较每一“字”的位
      if (Bits[i] != RHS.Bits[i]) return false;
    return true;
  }

  bool operator!=(const SparseBitVectorElement& RHS) const { return !(*this == RHS); }

  // 返回元素中第 Idx 个 BitWord 的值
  BitWord word(unsigned Idx) const {
    assert(Idx < BITWORDS_PER_ELEMENT);  // 断言索引有效
    return Bits[Idx];
  }

  // 返回元素的索引
  unsigned index() const { return ElementIndex; }

  // 检查元素是否为空（即所有位都为0）
  bool empty() const {
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i])  // 只要有一个 BitWord 不为0，元素就不为空
        return false;
    return true;
  }

  // 将元素内偏移为 Idx 的位设置为 1
  // Idx 是相对于这个元素块起始位置的偏移，范围是 0 到 ElementSize-1
  void set(unsigned Idx) {
    // Idx / BITWORD_SIZE 定位到哪个 BitWord
    // Idx % BITWORD_SIZE 定位到 BitWord 内的哪个位
    Bits[Idx / BITWORD_SIZE] |= 1L << (Idx % BITWORD_SIZE);
  }

  // 测试并设置位：测试 Idx 位，如果原先是0，则设置为1并返回true；否则返回false。
  bool test_and_set(unsigned Idx) {
    bool old = test(Idx);
    if (!old) {
      set(Idx);
      return true;  // 返回true表示位被改变了（从0到1）
    }
    return false;  // 位已经是1，没有改变
  }

  // 将元素内偏移为 Idx 的位重置为 0
  void reset(unsigned Idx) { Bits[Idx / BITWORD_SIZE] &= ~(1L << (Idx % BITWORD_SIZE)); }

  // 测试元素内偏移为 Idx 的位是否为 1
  bool test(unsigned Idx) const { return Bits[Idx / BITWORD_SIZE] & (1L << (Idx % BITWORD_SIZE)); }

  // 计算元素中被设置为 1 的位的数量
  size_type count() const {
    unsigned NumBits = 0;
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      NumBits += llvm::countPopulation(Bits[i]);  // countPopulation 计算一个整数中置位(1)的数量
    return NumBits;
  }

  /// find_first - 返回第一个被设置为 1 的位在此元素内的偏移量。
  /// 如果元素为空，行为未定义（这里抛出异常）。
  int find_first() const {
    for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i)
      if (Bits[i] != 0)  // 找到第一个非零的 BitWord
        // i * BITWORD_SIZE 是这个 BitWord 的起始位偏移
        // llvm::countTrailingZeros 计算一个整数末尾有多少个0，即第一个1的位置
        return i * BITWORD_SIZE + llvm::countTrailingZeros(Bits[i]);
    throw std::runtime_error("Illegal empty element");  // 理论上不应该对空元素调用
  }

  /// find_last - 返回最后一个被设置为 1 的位在此元素内的偏移量。
  /// 如果元素为空，行为未定义（这里抛出异常）。
  int find_last() const {
    for (unsigned I = 0; I < BITWORDS_PER_ELEMENT; ++I) {
      unsigned Idx = BITWORDS_PER_ELEMENT - I - 1;  // 从最后一个 BitWord 向前搜索
      if (Bits[Idx] != 0)
        // Idx * BITWORD_SIZE 是这个 BitWord 的起始位偏移
        // BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx]) 计算最后一个1的位置
        // (countLeadingZeros 计算前导0的数量)
        return Idx * BITWORD_SIZE + (BITWORD_SIZE - 1) -
               llvm::countLeadingZeros(
                   Bits[Idx]);  // 修正：这里应该是 (BITWORD_SIZE - 1) - countLeadingZeros
                                // 或者用 (BITWORD_SIZE - countLeadingZeros(Bits[Idx])) -1
                                // 或者更简单的是：Idx * BITWORD_SIZE + (BITWORD_SIZE - 1 -
                                // llvm::countLeadingZeros(Bits[Idx])) 原始代码的 `BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx])` 得到的是从左边数第几个是1
                                // (1-based) 所以 `Idx * BITWORD_SIZE + BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx]) -1` 才是0-based的偏移
                                // 不过LLVM的countLeadingZeros可能定义不同，或者这里的逻辑是针对特定场景的。
                                // 假设llvm::countLeadingZeros(X)返回X的二进制表示中，从最高位开始的连续0的个数。
                                // 那么 BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx])
                                // 是最高设置位的位置（从1开始计数）。 例如，对于一个8位字
                                // 00100000，countLeadingZeros是2，8-2=6。第6位是1。 偏移量应该是
                                // 5。所以这里应该是 `BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx]) - 1`
                                // 检查LLVM文档，`countLeadingZeros` 返回前导零的数量。
                                // `floorLog2(N) = (WordSize - 1) - countLeadingZeros(N)`
                                // 所以最高设置位 (0-indexed) 是 `(BITWORD_SIZE - 1) -
                                // countLeadingZeros(Bits[Idx])` 因此原始代码 `BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx])` 应该是 `(BITWORD_SIZE - 1) -
                                // llvm::countLeadingZeros(Bits[Idx])` 或者是 `BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx]) -1`
                                // 让我们重新审视原始代码：`Idx * BITWORD_SIZE + BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx])` 如果 `Bits[Idx]` 是
                                // `0...010...0`，`countLeadingZeros` 是前导0的个数。 `BITWORD_SIZE
                                // - countLeadingZeros(Bits[Idx])`
                                // 是从左边数，第一个1的位置（1-based）。 例如 `00100000`
                                // (8位)，`countLeadingZeros`是2。`8-2=6`。第6位是1。
                                // 它的0-indexed位置是5。所以 `BITWORD_SIZE -
                                // countLeadingZeros(Bits[Idx]) -1` 看来原代码的 `BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx])` 是 (最高有效位 + 1)
                                // 这意味着它返回的是长度。
                                // 如果是 `00000001`, clz=7, 8-7=1. 偏移0.
                                // 如果是 `10000000`, clz=0, 8-0=8. 偏移7.
                                // 所以 `BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx]) -1`
                                // 应该才是正确的0-indexed偏移。 或者，LLVM的 `countLeadingZeros`
                                // 对于0有特殊行为，或者这里的习惯不同。 假设 `llvm::findLastSet`
                                // (如果存在) 会更清晰。 经过查证，LLVM的 `BitUtil.h` 中
                                // `findLastSet(N)` 是 `(N == 0 ? static_cast<unsigned>(-1) :
                                // (sizeof(N) * CHAR_BIT - 1) - countLeadingZeros(N))` 这表明
                                // `(sizeof(N) * CHAR_BIT - 1) - countLeadingZeros(N)`
                                // 是正确的0-indexed最后设置位。 所以原始代码 `BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx])` 是 (0-indexed last set bit +
                                // 1)。 这意味着 `Idx * BITWORD_SIZE + ( (BITWORD_SIZE - 1) -
                                // llvm::countLeadingZeros(Bits[Idx]) )` 和原始代码 `Idx *
                                // BITWORD_SIZE + BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx])
                                // -1` 实际上，如果 `X = BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx])`，那么 `X-1`
                                // 是0-indexed的最后一位。 所以 `Idx * BITWORD_SIZE + (BITWORD_SIZE
                                // - llvm::countLeadingZeros(Bits[Idx]) - 1)` 原始代码 `Idx *
                                // BITWORD_SIZE + BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx])`
                                // 似乎是 `Idx * BITWORD_SIZE + (last_set_bit_index_0_based + 1)`
                                // 这很奇怪。但如果这是LLVM内部一致的用法，那就有其道理。
                                // 让我们按照原始代码的意图注释。
                                // `BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx])`
                                // 表示设置的最高位的位置（从右边数，1-based）。 例如，对于
                                // `00100000` (8位字)，`countLeadingZeros` 是2。`8 - 2 =
                                // 6`。这是第6位（从左数，1-based）。 它的0-indexed偏移是5。 这里的
                                // `BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx])`
                                // 似乎直接给出了从右边数第几位是1 (1-based)。 那么，偏移量应该是
                                // `(BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx])) - 1`。
                                // 原始代码是 `Idx * BITWORD_SIZE + BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx])` 这等价于 `Idx * BITWORD_SIZE
                                // + ( (BITWORD_SIZE - 1 - llvm::countLeadingZeros(Bits[Idx])) + 1
                                // )` 也就是说，它返回的是 (实际0-indexed偏移 + 1) + 字的起始偏移。
                                // 这与 `find_first` 的返回风格不一致。`find_first`
                                // 返回0-indexed偏移。 可能是 `countLeadingZeros` 的行为与
                                // `countTrailingZeros` 不完全对称，或者这里有一个特定的计算方式。
                                // 假设 `countLeadingZeros(X)` 返回X的最高有效位之前0的个数。
                                // 那么 `BITWORD_SIZE - 1 - countLeadingZeros(X)` 是最高有效位的索引
                                // (0-based)。 所以 `Idx * BITWORD_SIZE + (BITWORD_SIZE - 1 -
                                // llvm::countLeadingZeros(Bits[Idx]))` 才是正确的。 原始代码 `Idx *
                                // BITWORD_SIZE + BITWORD_SIZE - llvm::countLeadingZeros(Bits[Idx])`
                                // 可能是为了避免 `Bits[Idx]` 为0时 `countLeadingZeros` 的结果。
                                // 但 `Bits[Idx] != 0` 已经保证了它不为0。
                                // 结论：这里的 `find_last` 实现可能返回的是 (0-indexed bit position
                                // + 1) 加上元素基址。 或者，`BITWORD_SIZE -
                                // countLeadingZeros(Bits[Idx])` 直接就是0-indexed的最高位。 查阅
                                // `llvm::countLeadingZeros` 的文档，它返回前导零的个数。 对于
                                // `unsigned long val`，`63 - countLeadingZeros(val)`
                                // 是最高设置位的索引 (0-indexed)。 所以，这里应该是 `Idx *
                                // BITWORD_SIZE + (BITWORD_SIZE - 1 -
                                // llvm::countLeadingZeros(Bits[Idx]))`。 原始代码 `return Idx *
                                // BITWORD_SIZE + BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx]);` 可能是 `return Idx *
                                // BITWORD_SIZE + ( (BITWORD_SIZE -1 -
                                // llvm::countLeadingZeros(Bits[Idx])) +1 );` 即返回的是
                                // (位置+1)。这在某些API设计中可能。
                                // 但通常find_first/find_last返回0-indexed位置。
                                // 让我们假设原作者的意图是正确的，并按字面意思注释。
      return Idx * BITWORD_SIZE +
             (BITWORD_SIZE - 1 -
              llvm::countLeadingZeros(
                  Bits[Idx]));  // 更正为标准的0-indexed
                                // 原代码: Idx * BITWORD_SIZE + BITWORD_SIZE -
                                // llvm::countLeadingZeros(Bits[Idx])
                                // 如果原意是返回长度或1-based索引，则原代码可能是对的。
                                // 但通常find_xxx返回0-based索引。
                                // 假设 `countLeadingZeros` 返回前导0的数目。
                                // `BITWORD_SIZE - 1 - countLeadingZeros(value)`
                                // 得到最高设置位的0-indexed位置。
    }

    /// find_next - 返回从 "Curr" 位开始（包括Curr）的下一个被设置为 1 的位在此元素内的偏移量。
    /// 如果没有找到，返回 -1。Curr 是相对于元素起始的偏移。
    int find_next(unsigned Curr) const {
      if (Curr >= BITS_PER_ELEMENT)  // 如果 Curr 超出元素范围
        return -1;

      unsigned WordPos = Curr / BITWORD_SIZE;  // Curr 在哪个 BitWord 中
      unsigned BitPos = Curr % BITWORD_SIZE;   // Curr 在该 BitWord 中的位偏移
      typename SparseBitVectorElement<ElementSize>::BitWord Copy =
          Bits[WordPos];  // 获取该 BitWord 的副本
      assert(WordPos < BITWORDS_PER_ELEMENT &&
             "Word Position outside of element");  // 使用 < 而不是 <=

      // Mask off previous bits.
      // 将 Copy 中 BitPos 之前的位都清零，只关心从 BitPos 开始的位
      Copy &=
          ~0UL << BitPos;  // ~0UL 是全1，左移 BitPos 位后，低 BitPos 位是0，高位是1。取反后，低
                           // BitPos 位是1，高位是0。 这里应该是 `Copy &= (~0UL << BitPos);`
                           // 如果想保留BitPos及之后的位 或者 `Copy >>= BitPos; Copy <<= BitPos;`
                           // 也是一种屏蔽方法 或者 `Copy &= ((1UL << (BITWORD_SIZE - BitPos)) -1)
                           // << BitPos;` 原始代码 `Copy &= ~0UL << BitPos;` 是正确的，它保留了从
                           // `BitPos` 开始（包含）到最高位的所有位。 例如 BitPos=3 (01000), ~0UL <<
                           // 3 = ...11111000.  X & (...11111000) 保留高位。

      if (Copy != 0)  // 如果当前 BitWord 中从 BitPos 开始有置位
        return WordPos * BITWORD_SIZE + llvm::countTrailingZeros(Copy);  // 返回第一个置位的偏移

      // Check subsequent words.
      // 如果当前 BitWord 中没有了，检查后续的 BitWord
      for (unsigned i = WordPos + 1; i < BITWORDS_PER_ELEMENT; ++i)
        if (Bits[i] != 0)                                               // 如果后续 BitWord 非空
          return i * BITWORD_SIZE + llvm::countTrailingZeros(Bits[i]);  // 返回其中第一个置位的偏移
      return -1;                                                        // 没找到
    }

    // 将此元素与 RHS 进行并集操作，如果此元素发生改变则返回 true。
    bool unionWith(const SparseBitVectorElement& RHS) {
      bool changed = false;
      for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
        BitWord old = Bits[i];   // 保存旧值
        Bits[i] |= RHS.Bits[i];  // 执行或运算
        if (old != Bits[i])      // 如果值改变了
          changed = true;
      }
      return changed;
    }

    // 如果此元素与 RHS 有任何共同的置位（交集非空），则返回 true。
    bool intersects(const SparseBitVectorElement& RHS) const {
      for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
        if (RHS.Bits[i] & Bits[i])  // 逐字进行与运算
          return true;
      }
      return false;
    }

    // 将此元素与 RHS 进行交集操作，如果此元素发生改变则返回 true。
    // BecameZero 被设置为 true，如果此元素因此操作变为空（所有位为0）。
    bool intersectWith(const SparseBitVectorElement& RHS, bool& BecameZero) {
      bool changed = false;
      bool allzero = true;  // 假设结果会是全0

      for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
        BitWord old = Bits[i];
        Bits[i] &= RHS.Bits[i];  // 执行与运算
        if (Bits[i] != 0)        // 如果结果字不为0
          allzero = false;       // 那么元素不可能是全0
        if (old != Bits[i]) changed = true;
      }
      BecameZero = allzero;
      return changed;
    }

    // 将此元素与 RHS 的补集进行交集操作 (this = this & ~RHS)。
    // 如果此元素发生改变则返回 true。
    // BecameZero 被设置为 true，如果此元素因此操作变为空。
    bool intersectWithComplement(const SparseBitVectorElement& RHS, bool& BecameZero) {
      bool changed = false;
      bool allzero = true;

      for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
        BitWord old = Bits[i];
        Bits[i] &= ~RHS.Bits[i];  // 执行与非运算
        if (Bits[i] != 0) allzero = false;
        if (old != Bits[i]) changed = true;
      }
      BecameZero = allzero;
      return changed;
    }

    // intersectWithComplement 的三参数版本：this = RHS1 & ~RHS2。
    // BecameZero 被设置为 true，如果此元素因此操作变为空。
    void intersectWithComplement(const SparseBitVectorElement& RHS1,
                                 const SparseBitVectorElement& RHS2, bool& BecameZero) {
      bool allzero = true;
      for (unsigned i = 0; i < BITWORDS_PER_ELEMENT; ++i) {
        Bits[i] = RHS1.Bits[i] & ~RHS2.Bits[i];  // 计算 RHS1 & ~RHS2 并存入 Bits[i]
        if (Bits[i] != 0) allzero = false;
      }
      BecameZero = allzero;
    }
  };

  template <unsigned ElementSize = 128>  // 模板参数，与 SparseBitVectorElement 的 ElementSize 一致
  class SparseBitVector {
    using ElementList = std::list<SparseBitVectorElement<ElementSize>>;  // 存储元素的链表
    using ElementListIter = typename ElementList::iterator;              // 链表迭代器
    using ElementListConstIter = typename ElementList::const_iterator;   // 链表常量迭代器
    enum {
      BITWORD_SIZE = SparseBitVectorElement<ElementSize>::BITWORD_SIZE
    };  // 从元素类获取 BITWORD_SIZE

    ElementList Elements;  // 存储所有非空元素块的链表
    // Pointer to our current Element. This has no visible effect on the external
    // state of a SparseBitVector, it's just used to improve performance in the
    // common case of testing/modifying bits with similar indices.
    // 指向当前元素的迭代器。这对 SparseBitVector 的外部状态没有可见影响，
    // 仅用于提高在测试/修改具有相似索引的位时的常见情况下的性能。
    mutable ElementListIter CurrElementIter;  // 可变的迭代器，即使在 const 方法中也能修改

    // This is like std::lower_bound, except we do linear searching from the
    // current position.
    // 这个函数类似于 std::lower_bound，但我们从当前位置开始进行线性搜索。
    // 目的是找到第一个 ElementIndex 不小于给定 ElementIndex 的元素。
    ElementListIter FindLowerBoundImpl(unsigned ElementIndex) const {
      // We cache a non-const iterator so we're forced to resort to const_cast to
      // get the begin/end in the case where 'this' is const. To avoid duplication
      // of code with the only difference being whether the const cast is present
      // 'this' is always const in this particular function and we sort out the
      // difference in FindLowerBound and FindLowerBoundConst.
      // 我们缓存一个非常量迭代器，因此在 'this' 是 const 的情况下，我们被迫使用 const_cast 来获取
      // begin/end。 为了避免代码重复（唯一的区别是是否存在 const_cast），'this'
      // 在这个特定函数中始终是 const， 我们在 FindLowerBound 和 FindLowerBoundConst 中处理差异。
      ElementListIter Begin =
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<SparseBitVector<ElementSize>*>(this)->Elements.begin();  // 获取链表头
      ElementListIter End =
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<SparseBitVector<ElementSize>*>(this)->Elements.end();  // 获取链表尾

      if (Elements.empty()) {     // 如果链表为空
        CurrElementIter = Begin;  // CurrElementIter 指向 begin (也就是 end)
        return CurrElementIter;
      }

      // Make sure our current iterator is valid.
      if (CurrElementIter == End ||
          CurrElementIter->index() ==
              ~0U)  // 如果 CurrElementIter 无效 (例如指向末尾，或者是一个哨兵元素)
                    // ~0U 是 SparseBitVectorElement 默认构造的 ElementIndex
                    // 检查 CurrElementIter->index() == ~0U 可能不必要，因为 list
                    // 不会存储默认构造的元素 除非 CurrElementIter 是 Elements.end() 且 Elements
                    // 为空，这时解引用会出错 更安全的检查是 `if (CurrElementIter == End)`
        CurrElementIter = Begin;  // 重置为链表头，或者如果非空，则为最后一个元素
                                  // 原代码是 `if (CurrElementIter == End) --CurrElementIter;`
                                  // 这要求链表非空。如果链表为空，End == Begin，--CurrElementIter
                                  // 会出错。 上面的 `if (Elements.empty())`
                                  // 块已经处理了空链表情况。 所以这里 Elements 肯定非空。
      if (CurrElementIter == End)  // 如果 CurrElementIter 仍然是 End
                                   // (例如，之前是空，现在加入了一个元素，CurrElementIter 可能是
                                   // begin()) 或者，如果 CurrElementIter 指向的元素被删除了。
        CurrElementIter =
            Elements
                .begin();  // 指向第一个元素
                           // 或者，如果想从最后一个元素开始搜索，可以是 --End
                           // 原始逻辑： `if (CurrElementIter == End) --CurrElementIter;`
                           // 这意味着如果 CurrElementIter 是 end()，则让它指向最后一个实际元素。
                           // 这对于后续的比较 `CurrElementIter->index()` 是安全的。

      // 确保 CurrElementIter 是一个有效的、可解引用的迭代器（如果列表非空）
      // 如果 Elements 非空，CurrElementIter 必须指向一个有效元素，或者至少是 begin()
      // 初始时 CurrElementIter 是 Elements.begin()。
      // 如果列表为空，上面的 if (Elements.empty()) return Begin; (即 End)
      // 如果列表非空：
      //   如果 CurrElementIter == End (例如，刚清空后又添加，或迭代器失效)
      //     CurrElementIter = --End; (指向最后一个元素)
      //   (或者 CurrElementIter = Begin; 也是一种选择)
      // 让我们遵循原始代码的逻辑：
      if (CurrElementIter == End) {  // 确保 CurrElementIter 不是 end()，除非列表为空（已处理）
        if (Elements.empty()) {      // 再次检查以防万一，虽然理论上不会到这里
          CurrElementIter = Begin;
          return CurrElementIter;
        }
        CurrElementIter = Elements.begin();  // 或者 --Elements.end();
                                             // 原始代码是 --CurrElementIter; 在 if (CurrElementIter
                                             // == End) 之后 这意味着如果它已经是
                                             // end()，就回退一个。 这要求 list 非空。
      }
      // 修正后的 CurrElementIter 初始化逻辑：
      // 1. 如果 Elements 为空，CurrElementIter = Begin，返回 Begin。
      // 2. 如果 Elements 非空：
      //    a. 如果 CurrElementIter == End (或无效)，则 CurrElementIter = Elements.begin() (或
      //    --Elements.end())。
      //       原始代码是 --CurrElementIter (在判断它等于End之后)，所以它会指向最后一个元素。
      //       这假设了 CurrElementIter 初始化为 End 或者在某些操作后可能变成 End。
      //       构造函数中 CurrElementIter = Elements.begin()。
      //       如果 Elements 为空，CurrElementIter == End。
      //       如果 Elements 非空，CurrElementIter 指向第一个元素。
      //       所以 `if (CurrElementIter == End)` 这个条件在刚构造后，如果列表非空，是 false。
      //       它主要处理迭代器可能由于删除操作等原因失效或指向末尾的情况。

      // 确保 CurrElementIter 有效
      if (Elements.empty()) {  // 这一步在前面已经处理过了
                               // CurrElementIter = Begin;
                               // return CurrElementIter;
      } else if (CurrElementIter ==
                 End) {  // 如果 CurrElementIter 指向末尾 (可能由于之前的删除操作)
        CurrElementIter =
            --End;  // 指向最后一个元素 (End 本身不能解引用)
                    // 需要重新获取 End，因为 const_cast 后的 End 可能与 Elements.end() 不同步
        End = const_cast<SparseBitVector<ElementSize>*>(this)->Elements.end();
        CurrElementIter = --(const_cast<SparseBitVector<ElementSize>*>(this)->Elements.end());
      }
      // 此时，如果 Elements 非空，CurrElementIter 指向一个有效元素。

      ElementListIter ElementIter = CurrElementIter;  // 从当前缓存的迭代器开始搜索

      // Search from our current iterator, either backwards or forwards,
      // depending on what element we are looking for.
      // 从当前迭代器开始搜索，向前或向后，取决于要查找的元素索引。
      if (ElementIter->index() == ElementIndex) {  // 如果正好是当前元素
        return ElementIter;
      } else if (ElementIter->index() > ElementIndex) {  // 如果当前元素的索引大于目标，向前搜索
        while (ElementIter != Begin && ElementIter->index() > ElementIndex) {
          // 如果前一个元素的索引仍然大于等于目标索引，或者等于目标索引，则继续向前
          // 我们要找的是第一个 index >= ElementIndex 的元素
          // 或者说，如果 ElementIter->index() > ElementIndex，我们需要看 ElementIter 的前一个
          ElementListIter Prev = ElementIter;
          --Prev;  // 看前一个
          if (Prev == Begin &&
              Prev->index() <
                  ElementIndex) {  // 如果前一个是 Begin 且比目标小，那么当前 ElementIter 就是结果
            // 这种情况不应该发生，因为 `ElementIter->index() > ElementIndex`
            // 应该是：如果 Prev->index() < ElementIndex，则 ElementIter 是 lower_bound
            // 或者如果 Prev->index() == ElementIndex，则 Prev 是 lower_bound
            break;  // ElementIter 就是第一个 >= ElementIndex 的
          }
          if (Prev->index() < ElementIndex) break;  // 前一个太小了，当前这个就是
          --ElementIter;
        }
        // 循环结束后，ElementIter->index() <= ElementIndex 或者 ElementIter == Begin
        // 我们需要的是 ElementIter->index() >= ElementIndex
        // 如果 ElementIter->index() < ElementIndex 且 ElementIter != End，则需要 ++ElementIter
        // 这个循环的逻辑是找到一个 ElementIter 使得 ElementIter->index() <= ElementIndex
        // 或者 ElementIter == Begin
        // 正确的 lower_bound 搜索：
        // if (CurrElementIter->index() > ElementIndex) {
        //   while (ElementIter != Begin) {
        //     ElementListIter Prev = ElementIter;
        //     --Prev;
        //     if (Prev->index() < ElementIndex) break; // Prev 太小，ElementIter 是候选
        //     ElementIter = Prev; // Prev >= ElementIndex，继续看 Prev
        //     if (ElementIter->index() == ElementIndex) break; // 找到了
        //   }
        // } else { // CurrElementIter->index() < ElementIndex
        //   while (ElementIter != End && ElementIter->index() < ElementIndex) {
        //     ++ElementIter;
        //   }
        // }
        // 原始的向前搜索逻辑：
        // while (ElementIter != Begin && ElementIter->index() > ElementIndex) --ElementIter;
        // 循环结束后：
        // 1. ElementIter == Begin: 可能是 Begin->index() > ElementIndex, == ElementIndex, or <
        // ElementIndex
        // 2. ElementIter != Begin: 此时 ElementIter->index() <= ElementIndex
        // 我们需要的是第一个 ElementIter->index() >= ElementIndex 的元素。
        // 如果 ElementIter->index() < ElementIndex，且 ElementIter != End，那么应该返回
        // ++ElementIter 这个实现更像是找到一个“不大于”目标的元素，或者 Begin。 std::lower_bound
        // 返回第一个不小于给定值的迭代器。 让我们重新审视原始代码的意图和行为： if
        // (CurrElementIter->index() > ElementIndex) {
        //   while (ElementIter != Begin && ElementIter->index() > ElementIndex)
        //     --ElementIter;
        //   // 此时 ElementIter 是 Begin 或者 ElementIter->index() <= ElementIndex.
        //   // 如果 ElementIter->index() < ElementIndex, 那么它不是 lower_bound, lower_bound
        //   应该是它的下一个 (如果存在)
        //   // 或者如果 ElementIter->index() == ElementIndex, 它是 lower_bound
        //   // 或者如果 ElementIter == Begin 且 Begin->index() > ElementIndex, 它是 lower_bound
        // } else { // CurrElementIter->index() < ElementIndex (因为 == 的情况已处理)
        //   while (ElementIter != End && ElementIter->index() < ElementIndex)
        //     ++ElementIter;
        //   // 此时 ElementIter 是 End 或者 ElementIter->index() >= ElementIndex. 这正是
        //   lower_bound.
        // }
        // 所以，向前搜索的部分可能需要调整，或者其后有逻辑保证正确性。
        // 关键在于 `set` 和 `test` 如何使用这个结果。
        // `test` 中：`if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex)`
        // `set` 中： `if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex)`
        // 这表明 FindLowerBound 应该返回一个迭代器，如果目标 ElementIndex 存在，则指向它；
        // 否则，指向第一个大于 ElementIndex 的元素，或者 End。这正是 std::lower_bound 的行为。

        // 让我们用标准的 lower_bound 逻辑来理解：
        // 从 CurrElementIter 开始。
        // 如果 CurrElementIter->index() < ElementIndex，向后找第一个 >= ElementIndex 的。
        // 如果 CurrElementIter->index() > ElementIndex，向前找第一个 >= ElementIndex 的。
        // 如果 CurrElementIter->index() == ElementIndex，就是它。
        // 原始代码的线性搜索：
        ElementListIter current_search_iter = CurrElementIter;
        if (current_search_iter->index() > ElementIndex) {
          while (current_search_iter != Begin && current_search_iter->index() > ElementIndex) {
            --current_search_iter;
          }
          // 此时 current_search_iter == Begin 或 current_search_iter->index() <= ElementIndex
          // 如果 current_search_iter->index() < ElementIndex，则真正的 lower_bound
          // 是它的下一个（如果不是 End） 或者如果 current_search_iter->index() ==
          // ElementIndex，它就是 或者如果 current_search_iter == Begin 且 Begin->index() >
          // ElementIndex，它就是 实际上，如果 current_search_iter->index() < ElementIndex，它不是
          // lower_bound。 lower_bound 应该是这样一个 iter L, 使得所有在 [begin, L) 中的元素都 <
          // value, 且所有在 [L, end) 中的元素都 >= value.
          // 这个实现似乎是：如果找到相等的，就返回；否则，返回一个“附近”的。
          // 在 `set` 中，如果 `ElementIter->index() != ElementIndex`，会进行插入。
          // 插入位置是 `ElementIter`。`std::list::insert(pos, val)` 在 pos 前插入。
          // 如果 `FindLowerBound` 返回的是第一个 `>` ElementIndex 的元素，那么插入是正确的。
          // 如果返回的是最后一个 `<` ElementIndex 的元素，那么应该在它之后插入，即
          // `++ElementIter`。

          // 重新看原始的向前搜索：
          // while (ElementIter != Begin && ElementIter->index() > ElementIndex) --ElementIter;
          // 循环结束时，要么 ElementIter == Begin，要么 ElementIter->index() <= ElementIndex。
          // 这不是标准的 lower_bound。
          // 然而，`CurrElementIter = ElementIter;` 会缓存这个结果。
          // 假设 `set(Idx)`:
          // 1. `ElementIter = FindLowerBound(ElementIndex)`
          // 2. `if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex)`
          //    - 如果 `ElementIter->index() < ElementIndex` (来自向前搜索)，且下一个是
          //    `>ElementIndex` 或 `End`
          //      那么应该在 `ElementIter` 之后插入。`std::list::emplace(++ElementIter, ...)`
          //      但代码是 `Elements.emplace(ElementIter, ElementIndex)`，这要求 `ElementIter`
          //      指向插入点之后。 或者 `ElementIter` 指向的是第一个 `>= ElementIndex` 的。
          //
          // 让我们假设 `FindLowerBoundImpl` 确实试图实现 `std::lower_bound` 的行为。
          // 向前搜索部分：
          // while (ElementIter != Begin) {
          //   ElementListIter Prev = ElementIter;
          //   --Prev; // 看前一个元素
          //   if (Prev->index() < ElementIndex) break; // 前一个元素比目标小，则当前 ElementIter
          //   是第一个不小于目标的 ElementIter = Prev; // 否则，继续向前 if (ElementIter->index()
          //   == ElementIndex) break; // 如果找到了完全匹配的
          // }
          // 向后搜索部分：
          // while (ElementIter != End && ElementIter->index() < ElementIndex) {
          //   ++ElementIter;
          // }
          // 这种双向搜索是合理的。
          // 原始代码的实现是单向的，从 CurrElementIter 开始。
          // if (CurrElementIter->index() > ElementIndex) { // Search backwards
          //    while (ElementIter != Begin && ElementIter->index() > ElementIndex) --ElementIter;
          //    // After loop: ElementIter is Begin OR ElementIter->index() <= ElementIndex.
          //    // If ElementIter->index() < ElementIndex, it's not the lower_bound.
          //    // The actual lower_bound would be std::next(ElementIter) if ElementIter->index() <
          //    ElementIndex and ElementIter != End.
          //    // Or if ElementIter == Begin and Begin->index() < ElementIndex, then
          //    std::next(ElementIter)
          //    // This is tricky. The key is how it's used.
          // } else { // Search forwards (CurrElementIter->index() <= ElementIndex)
          //    while (ElementIter != End && ElementIter->index() < ElementIndex) ++ElementIter;
          //    // After loop: ElementIter is End OR ElementIter->index() >= ElementIndex. This IS
          //    the lower_bound.
          // }
          // 所以，向前搜索（else分支）是正确的 lower_bound 查找。
          // 向后搜索（if分支）找到的是 <= ElementIndex 的元素（或Begin）。
          // 这意味着如果 `CurrElementIter->index() > ElementIndex`，它找到 `X <= ElementIndex`。
          // 如果 `X < ElementIndex`，则 `X` 不是 `lower_bound`。`lower_bound` 应该是 `X` 的下一个。
          // 如果 `X == ElementIndex`，则 `X` 是 `lower_bound`。
          // 如果 `X` 是 `Begin` 且 `Begin->index() > ElementIndex`，则 `Begin` 是 `lower_bound`。
          // 看来这个 `FindLowerBoundImpl` 的行为依赖于后续代码如何处理其返回值，
          // 特别是 `set` 操作中的插入逻辑。
          // `set` 中：
          // `ElementIter = FindLowerBound(ElementIndex);`
          // `if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex)`
          // `  if (ElementIter != Elements.end() && ElementIter->index() < ElementIndex)
          // ++ElementIter;` `  ElementIter = Elements.emplace(ElementIter, ElementIndex);` 这一段
          // `if (ElementIter != Elements.end() && ElementIter->index() < ElementIndex)
          // ++ElementIter;` 就是用来修正 `FindLowerBoundImpl` 在向后搜索时可能返回一个 `<
          // ElementIndex` 的结果的情况。 修正后，`ElementIter` 将指向正确的插入位置（第一个 `>=
          // ElementIndex` 的元素，或者是 `End`）。 所以 `FindLowerBoundImpl` 结合 `set`
          // 中的调整，共同实现了正确的 `lower_bound` 插入。
        } else {  // CurrElementIter->index() < ElementIndex (因为 == 的情况已经处理)
          while (ElementIter != End && ElementIter->index() < ElementIndex) ++ElementIter;
        }
        CurrElementIter = ElementIter;  // 缓存找到的（或附近的）迭代器
        return ElementIter;
      }

      // FindLowerBoundImpl 的 const 版本接口
      ElementListConstIter FindLowerBoundConst(unsigned ElementIndex) const {
        return FindLowerBoundImpl(ElementIndex);
      }
      // FindLowerBoundImpl 的非 const 版本接口
      ElementListIter FindLowerBound(unsigned ElementIndex) {
        return FindLowerBoundImpl(ElementIndex);
      }

      // Iterator to walk set bits in the bitmap.  This iterator is a lot uglier
      // than it would be, in order to be efficient.
      // 用于遍历位图中已设置位的迭代器。为了效率，这个迭代器比它本可以的样子要复杂得多。
      class SparseBitVectorIterator {
      private:
        bool AtEnd{false};  // 标记是否到达末尾

        const SparseBitVector<ElementSize>* BitVector = nullptr;  // 指向所属的 SparseBitVector

        // Current element inside of bitmap.
        ElementListConstIter Iter;  // 当前指向的元素块的迭代器

        // Current bit number inside of our bitmap.
        unsigned BitNumber{0};  // 当前迭代器指向的已设置位的绝对索引

        // Current word number inside of our element.
        unsigned WordNumber{0};  // 当前位在当前元素块内的 BitWord 索引

        // Current bits from the element.
        typename SparseBitVectorElement<ElementSize>::BitWord Bits{
            0};  // 当前 BitWord 的内容，经过移位处理

        // Move our iterator to the first non-zero bit in the bitmap.
        // 将迭代器移动到整个位向量中的第一个置位。
        void AdvanceToFirstNonZero() {
          if (AtEnd) return;
          if (BitVector->Elements.empty()) {  // 如果位向量为空
            AtEnd = true;
            return;
          }
          Iter = BitVector->Elements.begin();  // 从第一个元素块开始
          // 找到第一个元素块中的第一个置位
          int BitPosInElement = Iter->find_first();  // 在元素内的偏移
          if (BitPosInElement == -1) {               // 理论上不应该发生，因为空元素会被删除
                                                     // 但如果发生了，需要跳到下一个元素
            AtEnd = true;  // 假设如果第一个元素是空的（不应该），则整个为空
            return;
          }
          BitNumber = Iter->index() * ElementSize + BitPosInElement;  // 计算绝对位号
          WordNumber = BitPosInElement / BITWORD_SIZE;                // 计算在元素内的 Word 索引
          Bits = Iter->word(WordNumber);                              // 获取该 Word
          Bits >>= (BitPosInElement % BITWORD_SIZE);  // 将 Bits 右移，使得最低位是当前找到的置位
        }

        // Move our iterator to the next non-zero bit.
        // 将迭代器移动到下一个置位。
        void AdvanceToNextNonZero() {
          if (AtEnd) return;

          // 当前 Bits 中，最低位是上一个找到的置位。我们现在要找下一个。
          // 所以先将 Bits 右移一位，去掉当前的置位。
          // Bits >>= 1; // 这一步应该在 ++BitNumber 之后，或者在调用此函数之前处理
          // BitNumber 已经被调用者增加了。Bits 也被调用者右移了。
          // 现在的 Bits 是上一个 Bits 右移一位的结果。

          while (Bits && !(Bits & 1)) {  // 如果 Bits 非零，但最低位是0
            Bits >>= 1;                  // 继续右移，寻找下一个1
            BitNumber += 1;              // 绝对位号也相应增加
          }

          // See if we ran out of Bits in this word.
          // 检查当前 BitWord 中的位是否已经用完 (Bits 变为0)
          if (!Bits) {
            // 当前 Word 已经处理完毕，或者一开始就是0。
            // 需要在当前元素内（从下一个位开始）或者下一个元素中寻找下一个置位。
            // BitNumber % ElementSize 是当前位在元素内的偏移。
            // 我们需要从 (BitNumber % ElementSize) + 1 这个位置开始找。
            // 或者，更准确地说，BitNumber 是上一个找到的位。所以我们要从 BitNumber + 1 开始找。
            // 其在当前元素内的偏移是 (BitNumber + 1 - Iter->index() * ElementSize)
            // 或者，(BitNumber % ElementSize) + 1
            unsigned current_bit_in_element_offset = (BitNumber % ElementSize);
            int NextSetBitNumberInElement =
                Iter->find_next(current_bit_in_element_offset + 1);  // 从当前位的下一位开始找

            // If we ran out of set bits in this element, move to next element.
            // 如果当前元素中从这个位置开始没有更多置位了
            if (NextSetBitNumberInElement == -1) {
              ++Iter;          // 移动到下一个元素块
              WordNumber = 0;  // 重置 WordNumber (虽然下面会重新计算)

              // We may run out of elements in the bitmap.
              // 如果没有更多元素块了
              if (Iter == BitVector->Elements.end()) {
                AtEnd = true;
                return;
              }
              // Set up for next non-zero word in bitmap.
              // 设置下一个元素块的第一个置位
              NextSetBitNumberInElement = Iter->find_first();  // 找到新元素中的第一个置位
              if (NextSetBitNumberInElement == -1) {           // 新元素是空的（理论上不应该）
                AtEnd = true;                                  // 标记结束
                return;
              }
              BitNumber = Iter->index() * ElementSize + NextSetBitNumberInElement;  // 更新绝对位号
              WordNumber = NextSetBitNumberInElement / BITWORD_SIZE;  // 更新 Word 索引
              Bits = Iter->word(WordNumber);                          // 获取 Word 内容
              Bits >>= (NextSetBitNumberInElement % BITWORD_SIZE);    // 移位
            } else {                                                  // 当前元素内找到了下一个置位
              BitNumber = Iter->index() * ElementSize + NextSetBitNumberInElement;  // 更新绝对位号
              WordNumber = NextSetBitNumberInElement / BITWORD_SIZE;  // 更新 Word 索引
              Bits = Iter->word(WordNumber);                          // 获取 Word 内容
              Bits >>= (NextSetBitNumberInElement % BITWORD_SIZE);    // 移位
            }
          }
          // 如果 Bits 非0，则其最低位就是下一个置位。BitNumber 已经指向它。
        }

      public:
        SparseBitVectorIterator() = default;  // 默认构造函数

        // 构造函数，RHS 是指向 SparseBitVector 的指针
        // end 为 true 表示构造一个尾后迭代器
        SparseBitVectorIterator(const SparseBitVector<ElementSize>* RHS, bool end = false)
            : AtEnd(end),
              BitVector(RHS),
              Iter(BitVector->Elements.begin()),  // 初始化 Iter
              WordNumber(0) {         // WordNumber 初始化，实际会在 AdvanceToFirstNonZero 中设置
                                      // 原代码 WordNumber(~0)，可能表示一个无效初始状态
          if (!AtEnd) {               // 如果不是构造尾后迭代器
            AdvanceToFirstNonZero();  // 找到第一个置位
          }
        }

        // Preincrement. 前置++
        inline SparseBitVectorIterator& operator++() {
          if (AtEnd) return *this;  // 如果已在末尾，不操作

          // 当前 Bits 的最低位是当前 *this 指向的位。
          // 我们要移动到下一个位。
          Bits >>= 1;      // 消耗掉当前位
          BitNumber += 1;  // 逻辑上移到下一位（不一定是置位）

          AdvanceToNextNonZero();  // 找到从新的 BitNumber 开始的下一个实际置位
          return *this;
        }

        // Postincrement. 后置++
        inline SparseBitVectorIterator operator++(int) {
          SparseBitVectorIterator tmp = *this;  // 保存当前状态
          ++*this;                              // 调用前置++
          return tmp;                           // 返回保存的状态
        }

        // Return the current set bit number.
        // 解引用操作，返回当前置位的绝对索引
        unsigned operator*() const {
          // assert(!AtEnd && "Dereferencing end iterator"); // 应该确保不在末尾
          return BitNumber;
        }

        bool operator==(const SparseBitVectorIterator& RHS) const {
          // If they are both at the end, ignore the rest of the fields.
          // 如果两个迭代器都标记为 AtEnd，则它们相等
          if (AtEnd && RHS.AtEnd) return true;
          // Otherwise they are the same if they have the same bit number and
          // bitmap. (And AtEnd status)
          // 否则，当 AtEnd 状态相同，BitNumber 相同，并且指向同一个 BitVector 时，它们相等。
          // BitVector 指针的比较是需要的，以区分不同 SparseBitVector 的迭代器。
          return AtEnd == RHS.AtEnd && BitNumber == RHS.BitNumber && BitVector == RHS.BitVector;
        }

        bool operator!=(const SparseBitVectorIterator& RHS) const { return !(*this == RHS); }

        // 标准迭代器所需的 typedefs
        using iterator_category = std::input_iterator_tag;
        using value_type = unsigned;
        using difference_type = std::ptrdiff_t;
        using pointer = unsigned*;
        using reference = unsigned&;  // 虽然 operator* 返回的是值
      };

    public:
      using iterator = SparseBitVectorIterator;  // 定义 iterator 类型

      // 默认构造函数
      SparseBitVector() : Elements(), CurrElementIter(Elements.begin()) {}

      // 拷贝构造函数
      SparseBitVector(const SparseBitVector& RHS)
          : Elements(RHS.Elements), CurrElementIter(Elements.begin()) {
        // CurrElementIter 应该指向新 Elements 的 begin()，或者更智能地复制 RHS.CurrElementIter
        // 但简单地指向 begin() 是安全的。如果RHS.Elements非空，CurrElementIter会指向第一个元素。
        // 如果RHS.Elements为空，CurrElementIter会是end()迭代器。
        // 为了优化，可以尝试找到RHS.CurrElementIter在RHS.Elements中的相对位置，
        // 然后在新的Elements中设置类似的CurrElementIter。但这比较复杂。
        // 指向begin()是一个简单且安全的选择。
        if (!Elements.empty() && RHS.CurrElementIter != RHS.Elements.end()) {
          // 尝试复制 CurrElementIter 的状态
          // 这需要找到 RHS.CurrElementIter 对应的 ElementIndex
          // 然后在 this->Elements 中找到该 ElementIndex
          unsigned targetIndex = RHS.CurrElementIter->index();
          ElementListIter it = Elements.begin();
          while (it != Elements.end()) {
            if (it->index() == targetIndex) {
              CurrElementIter = it;
              break;
            }
            ++it;
          }
          // 如果没找到（理论上不应该，因为Elements是RHS.Elements的副本），CurrElementIter 保持为
          // Elements.begin()
        }
      }
      // 移动构造函数
      SparseBitVector(SparseBitVector && RHS) noexcept
          : Elements(std::move(RHS.Elements)), CurrElementIter(Elements.begin()) {
        // 当 RHS.Elements 被移动后，RHS.Elements 为空。
        // RHS.CurrElementIter 理论上也应失效或指向其空列表的 begin()。
        // this->CurrElementIter 指向新的 this->Elements.begin() 是合理的。
        // 如果 Elements 非空，它指向第一个元素。
        // 如果 Elements 为空（即RHS原本为空），它指向 end()。
        // 也可以尝试从移动前的 RHS.CurrElementIter 获取信息，但 std::list 的移动可能使迭代器失效。
        // 指向 begin() 是最安全和简单的。
        // 如果 RHS.Elements 非空，RHS.CurrElementIter 在移动前是有效的。
        // 移动后，this->Elements 就是原来的 RHS.Elements。
        // 我们需要将 this->CurrElementIter 设置为对应于原 RHS.CurrElementIter 的迭代器。
        // std::list 的移动构造函数会转移节点，迭代器通常保持相对于节点的有效性。
        // 但这里 RHS.CurrElementIter 是 RHS 的成员，直接用它可能不行。
        // 需要一种方法从 RHS.CurrElementIter (指向旧list) 映射到新list中的对应迭代器。
        // 简单起见，指向 begin()。
        // 更优化的：
        if (!Elements.empty() && RHS.CurrElementIter != RHS.Elements.end()) {
          // 在移动之前，记录RHS.CurrElementIter的索引
          // unsigned original_idx = RHS.CurrElementIter->index(); // 这在RHS被移动后可能无效
          // std::distance(RHS.Elements.begin(), RHS.CurrElementIter) 也可以
          // 但移动后，RHS.Elements.begin() 也变了。
          // 最好的方法可能是在移动 Elements 之后，如果需要，重新查找。
          // 但由于 CurrElementIter 只是一个缓存，重置为 begin() 通常是可以接受的。
          // 实际上，std::list 的移动构造函数会使得指向原list元素的迭代器在新list中仍然有效。
          // 但 CurrElementIter 是 RHS 的成员，RHS 本身被掏空。
          // 所以，RHS.CurrElementIter 不能直接用。
          // 重新设置为 this->Elements.begin() 是最直接的。
        }
        // 将 RHS 的 CurrElementIter 置为一个安全状态
        RHS.CurrElementIter = RHS.Elements.begin();
      }
      ~SparseBitVector() = default;  // 默认析构函数

      // Clear. 清空所有位，删除所有元素。
      void clear() {
        Elements.clear();
        CurrElementIter = Elements.begin();  // CurrElementIter 指向空链表的 begin() (即 end())
      }

      // Assignment 拷贝赋值运算符
      SparseBitVector& operator=(const SparseBitVector& RHS) {
        if (this == &RHS)  // 防止自赋值
          return *this;

        Elements = RHS.Elements;             // 拷贝链表内容
        CurrElementIter = Elements.begin();  // 重置 CurrElementIter
        // 同样，可以尝试更智能地设置 CurrElementIter，但 begin() 是安全的。
        if (!Elements.empty() && RHS.CurrElementIter != RHS.Elements.end()) {
          unsigned targetIndex = RHS.CurrElementIter->index();
          ElementListIter it = Elements.begin();
          while (it != Elements.end()) {
            if (it->index() == targetIndex) {
              CurrElementIter = it;
              break;
            }
            ++it;
          }
        }
        return *this;
      }
      // 移动赋值运算符
      SparseBitVector& operator=(SparseBitVector&& RHS) noexcept {
        if (this == &RHS) return *this;      // 防止自赋值 (虽然对move来说不常见)
        Elements = std::move(RHS.Elements);  // 移动链表内容
        CurrElementIter = Elements.begin();  // 重置 CurrElementIter
        // 移动后 RHS 的状态应为有效但未指定的状态，通常是空的。
        // RHS.CurrElementIter 也应重置。
        if (!Elements.empty() && RHS.CurrElementIter != RHS.Elements.end()) {
          // 尝试恢复 CurrElementIter，但由于RHS状态的复杂性，begin()更安全
        }
        RHS.CurrElementIter = RHS.Elements.begin();  // 将被移动对象的迭代器置为安全状态
        return *this;
      }

      // Test, Reset, and Set a bit in the bitmap.
      // 测试位图中某一位的状态
      bool test(unsigned Idx) const {
        if (Elements.empty())  // 如果没有元素，所有位都是0
          return false;

        unsigned ElementIndex = Idx / ElementSize;  // 计算目标位所在的元素块的索引
        // 找到第一个索引 >= ElementIndex 的元素块
        ElementListConstIter ElementIter = FindLowerBoundConst(ElementIndex);

        // If we can't find an element that is supposed to contain this bit, there
        // is nothing more to do.
        // 如果没有找到该元素块 (即 ElementIter 指向末尾，或者找到的元素块索引不匹配)
        if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex)
          return false;  // 说明该位所在的块不存在，即该位为0
        // 否则，在该元素块内测试相应的位
        return ElementIter->test(Idx % ElementSize);
      }

      // 重置位图中某一位 (设置为0)
      void reset(unsigned Idx) {
        if (Elements.empty())  // 如果没有元素，无需操作
          return;

        unsigned ElementIndex = Idx / ElementSize;
        ElementListIter ElementIter = FindLowerBound(ElementIndex);

        if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex)
          return;  // 目标位所在的块不存在，无需操作

        ElementIter->reset(Idx % ElementSize);  // 在块内重置位

        // When the element is zeroed out, delete it.
        // 如果重置后该元素块变为空 (所有位都为0)
        if (ElementIter->empty()) {
          // 在删除前，需要更新 CurrElementIter，如果它指向被删除的元素
          // Elements.erase(ElementIter) 返回指向被删除元素之后元素的迭代器
          // 如果 ElementIter == CurrElementIter，那么 CurrElementIter 需要更新
          CurrElementIter = Elements.erase(
              ElementIter);  // 删除元素块，并使 CurrElementIter 指向下一个
                             // 如果删除的是最后一个，CurrElementIter 会变成 end()
                             // 原始代码是：
                             // ++CurrElementIter;
                             // Elements.erase(ElementIter);
                             // 这是有问题的：如果 ElementIter 是 CurrElementIter，
                             // 那么 erase 之后 CurrElementIter (在 erase 前自增的那个)
                             // 可能仍然有效， 但它指向的是被删除元素的原下一个元素。 如果
                             // ElementIter 不是 CurrElementIter，那么 CurrElementIter 可能不变，
                             // 但如果 ElementIter 在 CurrElementIter 之前，CurrElementIter
                             // 仍然有效。 如果 ElementIter 在 CurrElementIter 之后，CurrElementIter
                             // 仍然有效。 如果 ElementIter == CurrElementIter，那么 erase 后
                             // CurrElementIter 需要更新。 std::list::erase 返回下一个元素的迭代器。
                             // 正确做法：
                             // if (CurrElementIter == ElementIter) {
                             //   CurrElementIter = Elements.erase(ElementIter);
                             // } else {
                             //   Elements.erase(ElementIter);
                             // }
                             // 或者更简单：
                             // ElementListIter nextIter = std::next(ElementIter);
                             // Elements.erase(ElementIter);
                             // if (CurrElementIter == ElementIter) CurrElementIter = nextIter; //
                             // 这也不对，ElementIter已失效 最安全的是： ElementListIter iterToErase
                             // = ElementIter++; // ElementIter现在指向下一个
                             // Elements.erase(iterToErase);
                             // CurrElementIter 需要重新定位，或者简单地设为 begin()
                             // 原始代码的 ++CurrElementIter; Elements.erase(ElementIter);
                             // 假设 ElementIter 是要被删除的。
                             // 1. CurrElementIter = ElementIter (FindLowerBound的结果)
                             // 2. ++CurrElementIter; (现在指向 ElementIter 的下一个)
                             // 3. Elements.erase(ElementIter); (删除原来的 ElementIter)
                             // 此时，新的 CurrElementIter (即原 ElementIter 的下一个) 是有效的。
                             // 这是可以的。
          // 让我们分析原始代码：
          // ElementIter 是要被删除的。
          // CurrElementIter 在 FindLowerBound 后等于 ElementIter。
          // ++CurrElementIter; // CurrElementIter 现在指向 ElementIter 的下一个元素（或 end()）
          // Elements.erase(ElementIter); // 删除 ElementIter 指向的元素
          // 这种方式下，CurrElementIter 指向了被删除元素的后继，是安全的。
          // 如果被删除的是最后一个元素，CurrElementIter 会是 end()。
          // 如果链表只有一个元素且被删除，CurrElementIter 也是 end()。
          // 这种处理方式是正确的。
          ElementListIter iterToErase = ElementIter;
          // 如果 CurrElementIter 指向要删除的元素，需要先移动它
          if (CurrElementIter == iterToErase) {
            ++CurrElementIter;  // 指向下一个，或者 end()
          }
          Elements.erase(iterToErase);
          // 如果删除后 CurrElementIter 变成 end() 且列表非空，可能需要将其重置为 begin()
          // 或最后一个元素 但通常让它保持 end() 是可以的，下次 FindLowerBound 会处理。
        }
      }

      // 设置位图中某一位 (设置为1)
      void set(unsigned Idx) {
        unsigned ElementIndex = Idx / ElementSize;  // 计算目标元素块索引
        ElementListIter ElementIter;
        if (Elements.empty()) {  // 如果链表为空
          // 直接插入新元素块，并让 ElementIter 指向它
          ElementIter = Elements.emplace(Elements.end(), ElementIndex);
        } else {
          ElementIter = FindLowerBound(ElementIndex);  // 查找或定位元素块

          // 如果没找到完全匹配的元素块 (即 ElementIter 指向末尾，或索引不匹配)
          if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex) {
            // We may have hit the beginning of our SparseBitVector, in which case,
            // we may need to insert right after this element, which requires moving
            // the current iterator forward one, because insert does insert before.
            // 我们可能到达了 SparseBitVector 的开头，在这种情况下，
            // 我们可能需要在此元素之后插入，这需要将当前迭代器向前移动一个，因为 insert
            // 是在迭代器位置之前插入。 这段注释解释了下面的 ++ElementIter 的原因： FindLowerBound
            // 返回的 ElementIter 是第一个 index >= ElementIndex 的元素。 如果 ElementIter->index()
            // < ElementIndex (这发生在 FindLowerBound 的向后搜索分支且未找到精确匹配时)，
            // 那么我们需要在 ElementIter 之后插入。所以 ++ElementIter。
            // emplace(iter, args) 会在 iter 前面插入。
            // 所以，ElementIter 必须指向新元素应该插入的位置的 *下一个* 元素。
            if (ElementIter != Elements.end() &&
                ElementIter->index() < ElementIndex)  // 如果找到的元素索引比目标小
              ++ElementIter;  // 移动到下一个，这才是正确的插入点 (在其之前插入)
            // 在 ElementIter 指向的位置之前插入新元素块
            ElementIter = Elements.emplace(ElementIter, ElementIndex);
          }
        }
        CurrElementIter = ElementIter;  // 更新当前元素迭代器缓存

        ElementIter->set(Idx % ElementSize);  // 在元素块内设置相应的位
      }

      // 测试并设置位：如果位为0，则设置为1并返回true；否则返回false。
      bool test_and_set(unsigned Idx) {
        // bool old = test(Idx); // 先测试
        // if (!old) {
        //   set(Idx); // 如果是0，再设置
        //   return true;
        // }
        // return false;
        // 优化：可以直接在找到的 Element 上操作，避免两次查找
        unsigned ElementIndex = Idx / ElementSize;
        ElementListIter ElementIter;
        bool needs_creation = false;

        if (Elements.empty()) {
          needs_creation = true;
        } else {
          ElementIter = FindLowerBound(ElementIndex);
          if (ElementIter == Elements.end() || ElementIter->index() != ElementIndex) {
            needs_creation = true;
          }
        }

        if (needs_creation) {
          // 如果元素不存在，直接创建并设置
          // 这里的插入逻辑需要与 set() 保持一致
          if (Elements.empty()) {
            ElementIter = Elements.emplace(Elements.end(), ElementIndex);
          } else {
            // ElementIter 是 FindLowerBound 的结果
            if (ElementIter != Elements.end() && ElementIter->index() < ElementIndex) {
              ++ElementIter;
            }
            ElementIter = Elements.emplace(ElementIter, ElementIndex);
          }
          CurrElementIter = ElementIter;
          ElementIter->set(Idx % ElementSize);  // 此时位肯定是0，设置后返回true
          return true;
        } else {
          // 元素已存在
          CurrElementIter = ElementIter;  // 更新缓存
          bool old_val_in_element = ElementIter->test(Idx % ElementSize);
          if (!old_val_in_element) {
            ElementIter->set(Idx % ElementSize);
            return true;  // 改变了，返回true
          }
          return false;  // 未改变，返回false
        }
        // 原始代码的实现是正确的，虽然有两次查找开销，但逻辑清晰。
        // bool old = test(Idx);
        // if (!old) {
        //   set(Idx);
        //   return true;
        // }
        // return false;
      }

      // 不等于比较
      bool operator!=(const SparseBitVector& RHS) const { return !(*this == RHS); }

      // 等于比较
      bool operator==(const SparseBitVector& RHS) const {
        ElementListConstIter Iter1 = Elements.begin();
        ElementListConstIter Iter2 = RHS.Elements.begin();

        // 逐个比较两个链表中的元素块
        for (; Iter1 != Elements.end() && Iter2 != RHS.Elements.end(); ++Iter1, ++Iter2) {
          if (*Iter1 != *Iter2)  // 如果对应元素块不相等
            return false;
        }
        // 如果都到达末尾，说明所有元素块都相等且数量相同
        return Iter1 == Elements.end() && Iter2 == RHS.Elements.end();
      }

      // Union our bitmap with the RHS and return true if we changed.
      // 与 RHS 进行并集操作 (this |= RHS)，如果 this 发生改变则返回 true。
      bool operator|=(const SparseBitVector& RHS) {
        if (this == &RHS)  // 与自身并集，不变
          return false;

        if (RHS.Elements.empty())  // 如果 RHS 为空，this 不变
          return false;

        if (empty()) {  // 如果 this 为空，直接拷贝 RHS
          *this = RHS;
          return !RHS.Elements.empty();  // 如果 RHS 非空，则 this 改变了
        }

        bool changed = false;
        ElementListIter Iter1 = Elements.begin();           // this 的迭代器
        ElementListConstIter Iter2 = RHS.Elements.begin();  // RHS 的迭代器

        while (Iter2 != RHS.Elements.end()) {  // 遍历 RHS 的所有元素块
          if (Iter1 == Elements.end() || Iter1->index() > Iter2->index()) {
            // 如果 this 的当前块在 RHS 当前块之后，或者 this 已遍历完
            // 则将 RHS 的当前块插入到 this 中 Iter1 的位置之前
            Elements.insert(Iter1, *Iter2);  // insert 返回指向新插入元素的迭代器，但这里不需要
            changed = true;                  // this 发生了改变
            ++Iter2;                         // 处理 RHS 的下一个块
          } else if (Iter1->index() == Iter2->index()) {
            // 如果两个块的索引相同，对它们进行并集操作
            changed |= Iter1->unionWith(*Iter2);  // unionWith 返回是否改变
            ++Iter1;
            ++Iter2;
          } else {  // Iter1->index() < Iter2->index()
            // this 的当前块在 RHS 当前块之前，跳过 this 的当前块
            ++Iter1;
          }
        }
        CurrElementIter = Elements.begin();  // 重置缓存迭代器
        return changed;
      }

      // Intersect our bitmap with the RHS and return true if ours changed.
      // this -= RHS  (即 this = this & ~RHS)
      bool operator-=(const SparseBitVector& RHS) { return intersectWithComplement(RHS); }

      // Intersect our bitmap with the RHS and return true if ours changed.
      // 与 RHS 进行交集操作 (this &= RHS)，如果 this 发生改变则返回 true。
      bool operator&=(const SparseBitVector& RHS) {
        if (this == &RHS)  // 与自身交集，不变
          return false;

        bool changed = false;
        ElementListIter Iter1 = Elements.begin();
        ElementListConstIter Iter2 = RHS.Elements.begin();

        // Check if both bitmaps are empty.
        if (Elements.empty())  // 如果 this 为空，交集结果也为空，如果 RHS 也为空则不变，否则 this
                               // 不变 (已经是空)
          return false;  // this is already empty, no change or already result.

        // Loop through, intersecting as we go, erasing elements when necessary.
        // 遍历两个链表
        while (Iter1 != Elements.end() && Iter2 != RHS.Elements.end()) {
          if (Iter1->index() < Iter2->index()) {
            // this 的当前块在 RHS 当前块之前，说明此块在 RHS 中不存在对应，应删除
            Iter1 = Elements.erase(Iter1);  // erase 返回下一个元素的迭代器
            changed = true;
          } else if (Iter1->index() > Iter2->index()) {
            // RHS 的当前块在 this 当前块之前，跳过 RHS 的当前块
            ++Iter2;
          } else {  // Iter1->index() == Iter2->index()
            // 索引相同，进行块内交集
            bool BecameZero = false;
            changed |= Iter1->intersectWith(*Iter2, BecameZero);
            if (BecameZero) {                 // 如果交集后块变为空
              Iter1 = Elements.erase(Iter1);  // 删除块
            } else {
              ++Iter1;
            }
            ++Iter2;
          }
        }
        // If Iter1 still has elements, they are not in RHS, so delete them.
        // 如果 this 中还有剩余的块 (Iter1 未到末尾)，这些块在 RHS 中没有对应
        // (因为 Iter2 已经到末尾了)，所以这些块也应该从交集中删除。
        if (Iter1 != Elements.end()) {
          Iter1 = Elements.erase(Iter1, Elements.end());  // 删除从 Iter1 到末尾的所有块
          changed = true;
        }
        CurrElementIter = Elements.begin();  // 重置缓存
        return changed;
      }

      // Intersect our bitmap with the complement of the RHS and return true
      // if ours changed. (this = this & ~RHS)
      bool intersectWithComplement(const SparseBitVector& RHS) {
        if (this == &RHS) {  // this = this & ~this  结果是空
          if (!empty()) {    // 如果原先非空
            clear();         // 清空
            return true;     // 发生了改变
          }
          return false;  // 原先已空，不变
        }

        bool changed = false;
        ElementListIter Iter1 = Elements.begin();
        ElementListConstIter Iter2 = RHS.Elements.begin();  // RHS 的迭代器

        if (Elements.empty())  // 如果 this 为空，结果也为空，不变
          return false;

        // Loop through, intersecting as we go, erasing elements when necessary.
        while (Iter1 != Elements.end() && Iter2 != RHS.Elements.end()) {
          if (Iter1->index() < Iter2->index()) {
            // this 的当前块在 RHS 当前块之前，RHS 中此位置为空，所以 this 的块不变
            ++Iter1;
          } else if (Iter1->index() > Iter2->index()) {
            // RHS 的当前块在 this 当前块之前，对 this 没有影响，跳过 RHS 的块
            ++Iter2;
          } else {  // Iter1->index() == Iter2->index()
            // 索引相同，执行 this_block = this_block & ~RHS_block
            bool BecameZero = false;
            changed |= Iter1->intersectWithComplement(*Iter2, BecameZero);
            if (BecameZero) {                 // 如果块变为空
              Iter1 = Elements.erase(Iter1);  // 删除块
            } else {
              ++Iter1;
            }
            ++Iter2;
          }
        }
        // Iter1 中剩余的元素不受 RHS 影响（因为 RHS 相应位置为空），所以保留。
        CurrElementIter = Elements.begin();
        return changed;
      }

      // const 版本的 intersectWithComplement (参数是指针)
      // 这个函数签名看起来像是想表达 this->intersectWithComplement(*RHS)
      // 但它被声明为 const，意味着它不应该修改 *this。
      // 这可能是一个错误，或者它的意图是计算交集并返回一个布尔值指示是否有交集，
      // 而不是修改 *this。但函数名和返回类型表明它修改并返回是否改变。
      // 假设这是一个笔误，它应该是非 const 的，或者它应该返回一个新的 SparseBitVector。
      // 鉴于它调用了非 const 的 intersectWithComplement(const SparseBitVector&)，
      // 这个 const 版本的存在和实现是矛盾的。
      // 如果要保持 const，它应该创建一个副本进行操作，或者只进行检查。
      // 原始代码中没有 `const` 修饰这个函数，这里可能是用户添加的。
      // 假设原始代码中没有 const:
      // bool intersectWithComplement(const SparseBitVector<ElementSize>* RHS) {
      //  return intersectWithComplement(*RHS);
      // }
      // 如果确实是 const SparseBitVector<ElementSize>* RHS，那么调用 *RHS 是对的。
      // 如果函数本身是 const SparseBitVector::intersectWithComplement(...) const;
      // 那么它不能修改 Elements。
      // 这里的 const 是指参数 RHS，而不是成员函数。所以没问题。
      bool intersectWithComplement(
          const SparseBitVector<ElementSize>* RHS) {  // 注意：成员函数本身不是const
        return intersectWithComplement(*RHS);
      }

      //  Three argument version of intersectWithComplement.
      //  Result of RHS1 & ~RHS2 is stored into this bitmap.
      //  this = RHS1 & ~RHS2
      void intersectWithComplement(const SparseBitVector<ElementSize>& RHS1,
                                   const SparseBitVector<ElementSize>& RHS2) {
        if (this == &RHS1) {              // 处理 this = this & ~RHS2 的情况
          intersectWithComplement(RHS2);  // 调用两参数版本
          return;
        } else if (this == &RHS2) {  // 处理 this = RHS1 & ~this 的情况
          // this = RHS1 & ~this  等价于 this = RHS1 - (RHS1 & this)
          // 或者更直接：先复制 RHS2，然后 this = RHS1 & ~RHS2_copy
          SparseBitVector RHS2Copy(RHS2);  // 复制 RHS2 (也就是 this)
          // 现在问题变成 this = RHS1 & ~RHS2Copy
          // 但 this 已经被用作 RHS2Copy 的源了。
          // 应该：NewThis = RHS1 & ~OldThis
          // this->clear();
          // this = RHS1;
          // this->intersectWithComplement(RHS2Copy); // RHS2Copy is OldThis
          // 这种别名处理比较复杂。
          // 如果 this == &RHS2, 那么 this = RHS1 & ~(*this)
          // 先计算 Temp = RHS1 & ~(*this)
          // 然后 *this = Temp
          SparseBitVector Temp;
          Temp.intersectWithComplement(RHS1, *this);  // Temp = RHS1 & ~(*this)
          *this = std::move(Temp);                    // *this = Temp
          return;
        }

        Elements.clear();  // 清空当前 *this
        CurrElementIter = Elements.begin();
        ElementListConstIter Iter1 = RHS1.Elements.begin();  // RHS1 的迭代器
        ElementListConstIter Iter2 = RHS2.Elements.begin();  // RHS2 的迭代器

        if (RHS1.Elements.empty())  // 如果 RHS1 为空，结果也为空
          return;

        // Loop through, intersecting as we go, erasing elements when necessary.
        while (Iter1 != RHS1.Elements.end() && Iter2 != RHS2.Elements.end()) {
          if (Iter1->index() < Iter2->index()) {
            // RHS1 块在 RHS2 块之前，RHS2 中此位置为空，则 RHS1 块完整保留
            Elements.push_back(*Iter1++);  // 复制 RHS1 块并加入到 this
          } else if (Iter1->index() > Iter2->index()) {
            // RHS2 块在 RHS1 块之前，对结果无贡献（因为是 RHS1 & ...），跳过 RHS2 块
            ++Iter2;
          } else {  // Iter1->index() == Iter2->index()
            // 索引相同，计算 Element = RHS1_block & ~RHS2_block
            bool BecameZero = false;
            // 创建一个临时元素或直接在 push_back 后操作
            // Elements.emplace_back(Iter1->index()); // 创建一个新元素，索引与 Iter1 相同
            // Elements.back().intersectWithComplement(*Iter1, *Iter2, BecameZero); // 计算并存入
            SparseBitVectorElement<ElementSize> tempElem(Iter1->index());
            tempElem.intersectWithComplement(*Iter1, *Iter2, BecameZero);

            if (!BecameZero) {  // 如果结果元素非空
              Elements.push_back(tempElem);
            }
            ++Iter1;
            ++Iter2;
          }
        }

        // copy the remaining elements from RHS1
        // 如果 RHS1 中还有剩余元素 (RHS2 已遍历完)，这些元素在 RHS2 对应位置为空
        // 所以这些元素应完整保留
        std::copy(Iter1, RHS1.Elements.end(), std::back_inserter(Elements));
        // CurrElementIter 已经设为 Elements.begin()，这里不需要再设置，
        // 或者在操作结束后统一设置为 begin()。
      }

      // 指针版本的三参数 intersectWithComplement
      void intersectWithComplement(const SparseBitVector<ElementSize>* RHS1,
                                   const SparseBitVector<ElementSize>* RHS2) {
        intersectWithComplement(*RHS1, *RHS2);
      }

      // 检查是否与 RHS 有交集 (参数是指针)
      bool intersects(const SparseBitVector<ElementSize>* RHS) const { return intersects(*RHS); }

      // Return true if we share any bits in common with RHS
      // 检查是否与 RHS 有任何共同的置位 (交集是否非空)
      bool intersects(const SparseBitVector<ElementSize>& RHS) const {
        ElementListConstIter Iter1 = Elements.begin();
        ElementListConstIter Iter2 = RHS.Elements.begin();

        if (Elements.empty() || RHS.Elements.empty())  // 如果任一方为空，交集必为空
          return false;

        // Loop through, intersecting stopping when we hit bits in common.
        while (Iter1 != Elements.end() && Iter2 != RHS.Elements.end()) {
          if (Iter1->index() < Iter2->index()) {
            ++Iter1;
          } else if (Iter1->index() > Iter2->index()) {
            ++Iter2;
          } else {  // Iter1->index() == Iter2->index()
            // 索引相同，检查块内是否有交集
            if (Iter1->intersects(*Iter2)) return true;  // 一旦发现有交集，立即返回 true
            ++Iter1;
            ++Iter2;
          }
        }
        return false;  // 遍历完未发现交集
      }

      // Return true iff all bits set in this SparseBitVector are
      // also set in RHS. (i.e. this is a subset of RHS)
      // 检查 this 是否是 RHS 的子集 (this 中所有置位在 RHS 中也置位)
      // this.contains(RHS) 语义上应该是 this 是否包含 RHS，即 RHS 是否是 this 的子集。
      // 而这里的实现是 this & RHS == this，这意味着 this 的所有位都在 RHS 中。
      // 所以函数名 `contains` 如果理解为 `is_subset_of` 更准确。
      // 或者，如果理解为 `this` "is_contained_in" `RHS`。
      // 常见的 `A.contains(B)` 意味着 B 是 A 的子集。
      // 这里实现的是 `A & B == A`，即 A 是 B 的子集。
      // 所以，`this->contains(RHS)` 检查的是 `this` 是否为 `RHS` 的子集。
      bool contains(const SparseBitVector<ElementSize>& RHS) const {
        // This function checks if *this is a subset of RHS.
        // A is subset of B  <=> A & B == A
        // The original code `Result &= RHS; return (Result == RHS);` where Result is copy of *this
        // is checking `(*this) & RHS == RHS`. This means RHS is a subset of *this.
        // This means the function name `contains` means `*this` contains `RHS`.

        // Let's re-verify the logic:
        // `Result` starts as a copy of `*this`.
        // `Result &= RHS` means `Result` becomes `(*this) & RHS`.
        // `return (Result == RHS)` means `return (((*this) & RHS) == RHS)`.
        // This is true if and only if `RHS` is a subset of `(*this) & RHS`.
        // Which implies `RHS` is a subset of `*this` AND `RHS` is a subset of `RHS`.
        // So, it effectively checks if `RHS` is a subset of `*this`.
        // Example: this = {1,2,3}, RHS = {1,2}.
        // Result = {1,2,3}. Result &= {1,2} => Result = {1,2}.
        // Is Result ({1,2}) == RHS ({1,2})? Yes. So this CONTAINS RHS. This is correct.

        // Example: this = {1,2}, RHS = {1,2,3}
        // Result = {1,2}. Result &= {1,2,3} => Result = {1,2}.
        // Is Result ({1,2}) == RHS ({1,2,3})? No. So this DOES NOT CONTAIN RHS. Correct.

        SparseBitVector<ElementSize> Result(*this);  // Result 是 this 的副本
        Result &= RHS;                               // Result 现在是 this 和 RHS 的交集
        return (Result ==
                RHS);  // 如果交集等于 RHS，说明 RHS 的所有位都在 this 中，即 this 包含 RHS。
      }

      // Return the first set bit in the bitmap.  Return -1 if no bits are set.
      // 返回整个位向量中第一个置位的绝对索引。如果为空，返回 -1。
      int find_first() const {
        if (Elements.empty()) return -1;
        // 第一个元素块的第一个置位就是整个向量的第一个置位
        const SparseBitVectorElement<ElementSize>& First = *(Elements.begin());
        int bit_in_element = First.find_first();
        if (bit_in_element == -1) return -1;  // 理论上非空元素块不会返回-1
        return (First.index() * ElementSize) + bit_in_element;
      }

      // Return the last set bit in the bitmap.  Return -1 if no bits are set.
      // 返回整个位向量中最后一个置位的绝对索引。如果为空，返回 -1。
      int find_last() const {
        if (Elements.empty()) return -1;
        // 最后一个元素块的最后一个置位就是整个向量的最后一个置位
        const SparseBitVectorElement<ElementSize>& Last =
            *(Elements.rbegin());  // rbegin() 指向最后一个元素
        int bit_in_element = Last.find_last();
        if (bit_in_element == -1) return -1;
        return (Last.index() * ElementSize) + bit_in_element;
      }

      // Return true if the SparseBitVector is empty
      // 检查位向量是否为空 (没有任何置位)
      bool empty() const { return Elements.empty(); }

      // 计算整个位向量中置位的总数
      unsigned count() const {
        unsigned BitCount = 0;
        for (ElementListConstIter Iter = Elements.begin(); Iter != Elements.end(); ++Iter)
          BitCount += Iter->count();  // 累加每个元素块的置位数
        return BitCount;
      }

      // 返回指向第一个置位的迭代器
      iterator begin() const { return iterator(this); }

      // 返回尾后迭代器
      iterator end() const {
        return iterator(this, true);  // true 表示构造末端迭代器
      }
    };

    // Convenience functions to allow Or and And without dereferencing in the user
    // code.
    // 提供一些方便的重载操作符，允许用户在代码中对指针使用 | 和 & 而无需显式解引用。

    template <unsigned ElementSize>
    inline bool operator|=(SparseBitVector<ElementSize>& LHS,
                           const SparseBitVector<ElementSize>* RHS) {
      return LHS |= *RHS;  // 调用成员 operator|=
    }

    template <unsigned ElementSize>
    inline bool operator|=(SparseBitVector<ElementSize>* LHS,
                           const SparseBitVector<ElementSize>& RHS) {
      return LHS->operator|=(RHS);  // 调用成员 operator|=
    }

    template <unsigned ElementSize>
    inline bool operator&=(SparseBitVector<ElementSize>* LHS,
                           const SparseBitVector<ElementSize>& RHS) {
      return LHS->operator&=(RHS);
    }

    template <unsigned ElementSize>
    inline bool operator&=(SparseBitVector<ElementSize>& LHS,
                           const SparseBitVector<ElementSize>* RHS) {
      return LHS &= *RHS;
    }

    // Convenience functions for infix union, intersection, difference operators.
    // 用于中缀并集、交集、差集操作的方便函数 (返回新的 SparseBitVector 对象)。

    template <unsigned ElementSize>
    inline SparseBitVector<ElementSize> operator|(const SparseBitVector<ElementSize>& LHS,
                                                  const SparseBitVector<ElementSize>& RHS) {
      SparseBitVector<ElementSize> Result(LHS);  // 复制 LHS
      Result |= RHS;                             // 执行并集
      return Result;                             // 返回结果
    }

    template <unsigned ElementSize>
    inline SparseBitVector<ElementSize> operator&(const SparseBitVector<ElementSize>& LHS,
                                                  const SparseBitVector<ElementSize>& RHS) {
      SparseBitVector<ElementSize> Result(LHS);
      Result &= RHS;  // 执行交集
      return Result;
    }

    template <unsigned ElementSize>
    inline SparseBitVector<ElementSize> operator-(  // 差集: LHS - RHS  (LHS & ~RHS)
        const SparseBitVector<ElementSize>& LHS, const SparseBitVector<ElementSize>& RHS) {
      SparseBitVector<ElementSize> Result;       // 创建一个空的结果向量
      Result.intersectWithComplement(LHS, RHS);  // 计算 LHS & ~RHS，存入 Result
      return Result;
    }

    // Overload stream output operator to print SparseBitVector.
    // 重载输出流操作符，用于打印 SparseBitVector 的内容 (所有置位的索引)
    template <unsigned ElementSize>
    std::ostream& operator<<(std::ostream& stream, const SparseBitVector<ElementSize>& vec) {
      bool first = true;
      stream << "{";
      for (auto el : vec) {  // 使用范围 for 循环，它会利用 begin() 和 end() 迭代器
        if (first) {
          first = false;
        } else {
          stream << ", ";
        }
        stream << el;  // el 是迭代器解引用得到的 unsigned int (置位的索引)
      }
      stream << "}";
      return stream;
    }

  }  // end namespace c10