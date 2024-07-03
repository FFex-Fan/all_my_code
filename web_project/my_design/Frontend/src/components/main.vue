<template>
  <div class="container">
    <el-row :gutter="5">
      <el-row>
        <el-tabs type="border-card">
          <el-tab-pane :label="label.des">
            <description></description>
          </el-tab-pane>
          <!--          <el-tab-pane :label="label.rank" :lazy="true">-->
          <!--            <rankchart></rankchart>-->
          <!--          </el-tab-pane>-->
        </el-tabs>
      </el-row>
    </el-row>
  </div>
</template>

<script>
import rankchart from "@/components/chart/rankchart";
import ojmessage from "@/components/utils/ojmessage";
import description from "@/components/utils/description";
import acrank from "@/components/utils/acrank";

export default {
  components: {
    rankchart,
    ojmessage,
    acrank,
    description,
  },
  name: "homepage",
  data() {
    return {
      label: {
        rank: "Rank",
        team: "Team",
        rule: "Rule",
        des: "Description"
      },

    };
  },
  mounted() {

  },
  methods: {}
};
</script>

<style scoped>
.el-row {
  margin-bottom: 12px;
}

</style>
