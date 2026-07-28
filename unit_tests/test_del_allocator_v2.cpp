#include <gtest/gtest.h>

#include "device/del_allocator_v2.hpp"
#include "device/device_manager.hpp"

using namespace tunx;

class DELAllocatorV2Test : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {
    DeviceManager& manager = DeviceManager::instance();
    for (const DeviceID& id : manager.get_all()) {
      Device& dev = manager.get(id);
      if (dev.device_type() == DeviceType::CPU) {
        device_ = &dev;
        break;
      }
    }
    ASSERT_NE(device_, nullptr) << "CPU device not found";
    allocator_ = DELAllocatorV2::create(*device_, nullptr);
  }

  void TearDown() override {
    if (allocator_) {
      // Clear hooks before clearing allocator if any
      allocator_->clear();
    }
  }

  Device* device_ = nullptr;
  std::shared_ptr<DELAllocatorV2> allocator_;
};

TEST_F(DELAllocatorV2Test, Initialization) {
  EXPECT_EQ(allocator_->reserved(), 0);
  EXPECT_EQ(allocator_->allocated(), 0);
  EXPECT_EQ(allocator_->side(), 0);
  EXPECT_EQ(allocator_->unused(), 0);
}

TEST_F(DELAllocatorV2Test, AllocateBasic) {
  dptr ptr = allocator_->allocate(100);
  EXPECT_NE(ptr.get(), nullptr);
  EXPECT_EQ(ptr.capacity(), 100);

  size_t aligned_size = align_up(100, DEFAULT_ALIGNMENT);
  EXPECT_EQ(allocator_->allocated(), aligned_size);
  EXPECT_GE(allocator_->reserved(), aligned_size);
  EXPECT_EQ(allocator_->unused(), allocator_->reserved() - aligned_size);
}

TEST_F(DELAllocatorV2Test, AllocateZeroBytes) {
  dptr ptr = allocator_->allocate(0);
  EXPECT_EQ(ptr.get(), nullptr);
  EXPECT_EQ(ptr.capacity(), 0);
  EXPECT_EQ(allocator_->allocated(), 0);
  EXPECT_EQ(allocator_->reserved(), 0);
}

TEST_F(DELAllocatorV2Test, ReclaimAndCoalesce) {
  size_t size1 = align_up(100, DEFAULT_ALIGNMENT);
  size_t size2 = align_up(200, DEFAULT_ALIGNMENT);
  size_t size3 = align_up(300, DEFAULT_ALIGNMENT);

  {
    dptr ptr1 = allocator_->allocate(100);
    dptr ptr2 = allocator_->allocate(200);
    dptr ptr3 = allocator_->allocate(300);

    EXPECT_EQ(allocator_->allocated(), size1 + size2 + size3);
  }
  // All ptrs are out of scope, they should be reclaimed.
  // Because they are all the active allocations, the empty slab should be merged/cleared partially
  // based on logic, but allocated() must definitely be 0.
  EXPECT_EQ(allocator_->allocated(), 0);
}

TEST_F(DELAllocatorV2Test, FlipAndSide) {
  EXPECT_EQ(allocator_->side(), 0);
  allocator_->flip();
  EXPECT_EQ(allocator_->side(), 1);
  allocator_->set_side(0);
  EXPECT_EQ(allocator_->side(), 0);
  EXPECT_THROW(allocator_->set_side(2), std::invalid_argument);
}

TEST_F(DELAllocatorV2Test, ReserveAndClear) {
  allocator_->ensure(1000);
  EXPECT_GE(allocator_->reserved(), 1000);
  EXPECT_EQ(allocator_->allocated(), 0);

  allocator_->clear();
  EXPECT_EQ(allocator_->reserved(), 0);
  EXPECT_EQ(allocator_->unused(), 0);
}

TEST_F(DELAllocatorV2Test, ClearWithActiveAllocations) {
  dptr ptr = allocator_->allocate(100);
  EXPECT_THROW(allocator_->clear(), std::runtime_error);
}

TEST_F(DELAllocatorV2Test, AllocationHooks) {
  size_t last_allocated = 0;
  auto hook = [&last_allocated](size_t allocated) { last_allocated = allocated; };

  size_t hook_id = allocator_->add_allocation_hook(hook);

  size_t aligned_size1 = align_up(100, DEFAULT_ALIGNMENT);
  dptr ptr = allocator_->allocate(100);
  EXPECT_EQ(last_allocated, aligned_size1);

  allocator_->remove_allocation_hook(hook_id);
  dptr ptr2 = allocator_->allocate(200);

  // The hook shouldn't be called for ptr2, so last_allocated should remain unchanged
  EXPECT_EQ(last_allocated, aligned_size1);
}

TEST_F(DELAllocatorV2Test, InstanceSingleton) {
  std::shared_ptr<DELAllocatorV2> inst1 = DELAllocatorV2::instance(*device_, nullptr);
  std::shared_ptr<DELAllocatorV2> inst2 = DELAllocatorV2::instance(*device_, nullptr);
  EXPECT_EQ(inst1.get(), inst2.get());
}
