<template>
  <div>
    <div class="el-row">
      <div class="toolbar">
        <div class="toolbar_left">
          <el-tabs type="border-card">
            <el-tab-pane label="题目描述">
              <el-col :span="12">
                <h1>LEFT</h1>
              </el-col>
              <el-col :span="12">
                <h1>RIGHT</h1>
              </el-col>
            </el-tab-pane>
            <el-tab-pane label="提交记录">
              #
            </el-tab-pane>
          </el-tabs>
        </div>

        <div class="toolbar_right">
          语言
          <div ></div>
        </div>
      </div>
    </div>

    <div class="el-row">

    </div>
  </div>
</template>

<script>

</script>

<style scoped>
.toolbar {
  //border-radius: 10px;
  //display: flex;
  //flex-direction: column;

}

.toolbar_left {
  //margin-top: auto;
}

.toolbar_right {
  position: relative;
  float: right;
  //margin-top: auto;
}
</style>
