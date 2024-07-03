<template>
  <div class="container">
    <el-tabs type="border-card"
             v-show="canshow"
             v-loading="loading">
      <el-tab-pane label="添加题目">
        <adminaddproblem></adminaddproblem>
      </el-tab-pane>

      <el-tab-pane label="添加选择题">
        <adminaddchoiceproblem></adminaddchoiceproblem>
      </el-tab-pane>

      <el-tab-pane label="添加比赛"
                   :lazy="true">
        <adminaddcontest></adminaddcontest>
      </el-tab-pane>
      <el-tab-pane label="题目列表"
                   :lazy="true">
        <adminchangepro></adminchangepro>
      </el-tab-pane>

      <el-tab-pane label="选择题列表"
                   :lazy="true">
        <adminchangechoiceproblem></adminchangechoiceproblem>
      </el-tab-pane>

      <el-tab-pane label="比赛列表"
                   :lazy="true">
        <adminchangecontest></adminchangecontest>
      </el-tab-pane>
      <el-tab-pane label="用户列表"
                   :disabled="!isadmin"
                   :lazy="true">
        <adminmanageuser></adminmanageuser>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script>
import adminaddproblem from "@/components/admin/adminaddproblem";
import adminaddcontest from "@/components/admin/adminaddcontest";
import adminchangepro from "@/components/admin/adminchangepro";
import adminchangecontest from "@/components/admin/adminchangecontest";
import adminmanageuser from "@/components/admin/adminmanageuser";
import adminsetting from "@/components/admin/adminsetting";
import adminaddchoiceproblem from "@/components/admin/adminaddchoiceproblem";
import adminchangechoiceproblem from "@/components/admin/adminchangechoiceproblem";

export default {
  name: "admin",
  components: {
    adminaddproblem,
    adminaddcontest,
    adminchangepro,
    adminchangecontest,
    adminmanageuser,
    adminsetting,
    adminaddchoiceproblem,
    adminchangechoiceproblem,
  },
  data() {
    return {
      type: 1,
      isadmin: false,
      canshow: false,
      loading: true
    };
  },
  methods: {},
  created() {
    this.type = parseInt(sessionStorage.type);
    if (this.type !== 2 && this.type !== 3) {
      // console.log("type: " + this.type);
      this.$message.error("您的权限不足，请联系管理员！");
      this.canshow = false;
      return;
    }
    this.canshow = true;
    if (this.type === 3) {
      this.isadmin = true;
    }
  },
  mounted() {
    this.loading = false
  },
};
</script>

<style scoped>
h1 {
  position: relative;
}
</style>
