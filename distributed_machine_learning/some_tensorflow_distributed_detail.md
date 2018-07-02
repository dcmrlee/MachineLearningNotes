# 本文重点介绍一些tensorflow的分布式相关的细节。（英文居多，少量中文解释）
- 重点参考的源码文件：tensorflow/python/training/distribute.py

## class DistributionStrategy
该类其实定义了分布式策略的接口，是非常重要的一个类，体现了tensorflow如何将分布式运行与模型之间解耦的做法。(changes can be hidden inside the specific layers
and other library classes that need special treatment to run in a distributed setting, so that most users' model definition code can
run unchanged.)

如果需要自行定义分布式策略，只需要实现该接口（继承类）即可.

### 一些high-level的概念
  - `数据并行`: run multiple copies of the modelon different slices of the input data
  - `tower`: one copy of the model, running on one slice of the input data.
  - `同步训练`: updates from each tower are aggregated together before updating the model variables
  - `异步训练`: each tower updates the model variables independently
  - `Parameter servers`: hosts that hold a single copy of parameters/variables.(一个或多个host, 一份参数)
  - `Mirrored variables`: variables that are copied to multiple devices, where we keep the copies in sync by applying the same updates to every copy.(多个设备上都保存的参数，且参数是相同的，设备之间会sync update)
  - `Reductions and Allreduce`: sync training, perform a reduction on the gradients to a parameter from each tower before applying the update, Allreduce is an algorithm for performing a reduction on values from multiple devices and making the result available on all of those devices.
  (*注：未来还会支持跨设备的变量partitioned variables)
  
### 基本用法
* Code written (as if) with no knowledge of class `DistributionStrategy`.
    This code should work as before, even if some of the layers, etc.
    used by that code are written to be distribution-aware. This is done
    by having a default `DistributionStrategy` that gives ordinary behavior,
    and by default being in a single tower context.
* Ordinary model code that you want to run using a specific
    `DistributionStrategy`. This can be as simple as:

    ```
    with my_distribution.scope():
      iterator = my_distribution.distribute_dataset(dataset)
      tower_train_ops = my_distribution.call_for_each_tower(
          tower_fn, iterator.get_next())
      train_op = tf.group(my_distribution.unwrap(tower_train_ops))
    ```

    This takes an ordinary `dataset` and `tower_fn` and runs it
    distributed using a particular `DistributionStrategy` in
    `my_distribution`. Any variables created in `tower_fn` are created
    using `my_distribution`'s policy, and library functions called by
    `tower_fn` can use the `get_tower_context()` API to get enhanced
    behavior in this case.

    Note that in the future we will add support for initializable
    Dataset iterators, at which point this example code will change.
* If you want to write a distributed algorithm, you may use any of
    the `DistributionStrategy` APIs inside a
    `with my_distribution.scope():` block of code.

### Lower-Level概念
  * Wrapped values: In order to represent values parallel across devices
    (either towers or the devices associated with a particular value), we
    wrap them in a "PerDevice" or "Mirrored" object that contains a map
    from device to values. "PerDevice" is used when the value may be
    different across devices, and "Mirrored" when the value are the same.
  * Unwrapping and merging: Consider calling a function `fn` on
    multiple devices, like `call_for_each_tower(fn, w)` with an
    argument `w` that is a wrapped value. This means `w` will have a
    map taking tower device `d0` to `w0`, tower device `d1` to `w1`,
    etc. `call_for_each_tower()` unwraps `w` before calling `fn`, so
    it calls `fn(w0)` on `d0`, `fn(w1)` on `d1`, etc.  It then merges
    the return values from `fn()`, which can possibly result in
    wrapped values. For example, let's say `fn()` returns a tuple with
    three components: `(x, a, v0)` from tower 0, `(x, b, v1)` on tower 1,
    etc. If the first component is the same object `x` from every
    tower, then the first component of the merged result will also be
    `x`. If the second component is different (`a`, `b`, ...)  from
    each tower, then the merged value will have a wrapped map from
    tower device to the different values. If the third component is
    the members of a mirrored variable (`v` maps `d0` to `v0`, `d1` to
    `v1`, etc.), then the merged result will be that mirrored variable
    (`v`).
  * Tower context vs. Cross-tower context: _tower context_ is when we
    are in some function that is being called once for each tower.
    Otherwise we are in cross-tower context, which is useful for
    calling `DistributionStrategy` methods which operate across the
    towers (like `reduce()`). By default you start in a tower context
    (the default "single tower context") and then some methods can
    switch you back and forth, as described below.
  * Worker devices vs. parameter devices: Most tower computations will
    happen on worker devices. Since we don't yet support model
    parallelism, there will be one worker device per tower. When using
    parameter servers (see above), the set of devices holding
    variables may be different, otherwise the parameter devices might
    match the worker devices.
  * Non-slot devices are some subset of the parameter devices where we
    put all the non-slot variables. We need to ensure that all
    non-slot variables are allocated on the same device, or mirrored
    across the same set of devices. If you have some variable you want
    to colocate all the non-slot variables with, you can use
    `colocate_vars_with()` to get the remaining non-slot variables on
    the same device.  Otherwise you can use `non_slot_devices()` to
    pick a consistent set of devices to pass to both
    `colocate_vars_with()` and `update_non_slot()`.

### Rules for methods with respect to locality and single-tower vs. cross-tower context
接口函数能体现看下解耦的思路
  * `with d.scope()`: default single-tower context -> cross-tower context for
    `d`
  * `with d.colocate_vars_with(v)`: in tower/cross-tower context, variables
    will be created with locality V(`v`). That is, if we write
    `with d.colocate_vars_with(v1): v2 = tf.get_variable(...)`, then
    `v2` will have locality V(`v1`), i.e. locality V(`v2`) will equal
    V(`v1`).
  * `with d.colocate_vars_with(d.non_slot_devices(...))`: in
    tower/cross-tower context, variables will be created with locality N
  * `v = tf.get_variable(...)`: in tower/cross-tower context, creates
    a variable (which by definition will have locality V(`v`), though
    will match another locality if inside a `colocate_vars_with`
    scope).
  * `d.distribute_dataset(dataset)`: in cross-tower context, produces an
    iterator with locality T
  * `d.broadcast(t)`: in cross-tower context, produces a value with locality M
  * `d.broadcast(t, v)`: in cross-tower context, produces a value with
    locality V(`v`)
  * `d.call_for_each_tower(fn, ...)`: in cross-tower context, runs
    `fn()` in a tower context (and so may call `get_tower_context()` and
    use its API, including `merge_call()` to get back to cross-tower
    context), once for each tower. May use values with locality T or
    M, and any variable.
  * `d.reduce(m, t)`: in cross-tower context, accepts t with locality T
    and produces a value with locality M.
  * `d.reduce(m, t, v)`: in cross-tower context, accepts t with
    locality T and produces a value with locality V(`v`).
  * `d.batch_reduce(m, [(t, v)])`: see `d.reduce()`
  * `d.update(v, fn, ...)`: in cross-tower context, runs `fn()` once
    for each device `v` is copied to, all inputs should have locality
    V(`v`), output will have locality V(`v`) as well.
  * `d.update_non_slot(d.non_slot_devices(), fn)`: in cross-tower
    context, like `d.update()` except with locality N.
  * `d.fetch(t)`: Copy `t` with any locality to the client's CPU device.

### standard pattern for updating variables
  1. Wrap your input dataset in `d.distribute_dataset()`.
  2. Define each tower `d.call_for_each_tower()` up to the point of
     getting a list of gradient, variable pairs.
  3. Call `d.reduce("sum", t, v)` or `d.batch_reduce()` to sum the
     gradients (with locality T) into values with locality V(`v`).
  4. Call `d.update(v)` for each variable to update its value.
  
  Steps 3 and 4 are done automatically by class `Optimizer` if you call
  its `apply_gradients` method in a tower context. Otherwise you can
  manually call its `_distributed_apply` method in a cross-tower context.
  
  Another thing you might want to do in the middle of your tower function
  is an all-reduce of some intermediate value, using `d.reduce()` or
  `d.batch_reduce()` without supplying a variable as the destination.
  
  Layers should expect to be called in a tower context, and can use
  the `get_tower_context()` function to get a `TowerContext` object.  The
  `TowerContext` object has a `merge_call()` method for entering
  cross-tower context where you can use `reduce()` (or
  `batch_reduce()`) and then optionally `update()` to update state.
 
### tensorflow参数同步代码的实例（ps、worker之间）
重点函数：call_for_each_tower
实例如下：
```python
    # Called once in "cross-tower" context.
    def merge_fn(distribution, three_plus_tower_id):
      # sum the values across towers
      return sum(distribution.unwrap(three_plus_tower_id))

    # Called once per tower in `distribution`, in a "tower" context.
    def fn(three):
      tower_ctx = tf.get_tower_context()
      v = three + tower_ctx.tower_id
      # Computes the sum of the `v` values across all towers.
      s = tower_ctx.merge_call(merge_fn, v)
      return s + v

    with distribution.scope():
      # in "cross-tower" context
      ...
      merged_results = distribution.call_for_each_tower(fn, 3)
      # merged_results has the values from every tower execution of `fn`.
      print(distribution.unwrap(merged_results))  # Prints a list
```
`fn` may call `tf.get_tower_context()` to access methods such as
    `tower_id()` and `merge_call()`.

 `merge_call()` is used to communicate betwen the towers and
    re-enter the cross-tower context. All towers pause their execution
    having encountered a `merge_call()` call. After that the
    `merge_fn`-function is executed. Its results are then unwrapped and
    given back to each tower call. After that execution resumes until
    `fn` is complete or encounters another `merge_call()`.
