<template>
  <el-card shadow="always">
    <h3>{{title}}</h3>
    <el-row :gutter="10" style="margin-top: 10px;">
      <el-col :span="3">
        <b>AC:</b>
      </el-col>
      <el-col :span="21">
        <el-progress :percentage="ac" color="#67C23A"></el-progress>
      </el-col>
    </el-row>

    <el-row :gutter="10">
      <el-col :span="3">
        <b>WA:</b>
      </el-col>
      <el-col :span="21">
        <el-progress :percentage="wa" color="#F56C6C"></el-progress>
      </el-col>
    </el-row>
    <el-row :gutter="10">
      <el-col :span="3">
        <b>PE:</b>
      </el-col>
      <el-col :span="21">
        <el-progress :percentage="pe" color="#FF9800"></el-progress>
      </el-col>
    </el-row>
    <el-row :gutter="10">
      <el-col :span="3">
        <b>TLE:</b>
      </el-col>
      <el-col :span="21">
        <el-progress :percentage="tle" color="#FF9800"></el-progress>
      </el-col>
    </el-row>
    <el-row :gutter="10">
      <el-col :span="3">
        <b>RTE:</b>
      </el-col>
      <el-col :span="21">
        <el-progress :percentage="rte" color="#795548"></el-progress>
      </el-col>
    </el-row>
    <el-row :gutter="10">
      <el-col :span="3">
        <b>MLE:</b>
      </el-col>
      <el-col :span="21">
        <el-progress :percentage="mle" color="#795548"></el-progress>
      </el-col>
    </el-row>

    <el-row :gutter="10">
      <el-col :span="3">
        <b>CE:</b>
      </el-col>
      <el-col :span="21">
        <el-progress :percentage="ce" color="#FFC107"></el-progress>
      </el-col>
    </el-row>

    <el-row :gutter="10">
      <el-col :span="3">
        <b>SE:</b>
      </el-col>
      <el-col :span="21">
        <el-progress :percentage="se" color="#FFC107"></el-progress>
      </el-col>
    </el-row>
  </el-card>
</template>

<script>
export default {
  name: "prostatistice",
  data() {
    return {
      ac: 100,
      mle: 100,
      tle: 100,
      rte: 100,
      pe: 100,
      ce: 100,
      wa: 100,
      se: 100,
      title: "",
    };
  },
  created() {
  },
  methods: {
    setdata(d){
        this.ac=d.ac
        this.mle=d.mle
        this.tle=d.tle
        this.rte=d.rte
        this.pe=d.pe
        this.ce=d.ce
        this.wa=d.wa
        this.se=d.se
        this.title=d.title
    }
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
#leveltag {
  text-align: center;
  font-weight: bold;
}
.el-row {
  margin-bottom: 20px;
}
</style>
