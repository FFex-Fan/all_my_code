<template>
  <el-row :gutter="15">
    <el-col :span="5" style="margin-left: 20px">
      <el-row :gutter="15">
        <el-col>
          <prostatistice ref="prosta"></prostatistice>
        </el-col>
      </el-row>
      <el-row>
        <el-card shadow="always">
          <el-input placeholder="Search..."
                    v-model="searchtext"
                    @keyup.native.enter="searchtitle">
            <el-button slot="append"
                       icon="el-icon-search"
                       @click="searchtitle"></el-button>
          </el-input>
        </el-card>
      </el-row>
      <el-row :gutter="15">
        <el-col>
          <el-card shadow="always">
            <h5>Tags</h5>
            <el-button id="tag"
                       v-for="(name,index) in tagnames"
                       :key="index"
                       size="mini"
                       style="margin-top: 10px"
                       @click="tagclick(name)"
                       :ref="name">{{ name }}
            </el-button>
          </el-card>
        </el-col>
      </el-row>
    </el-col>

    <el-col :span="18">
      <el-card shadow="always">
        <el-switch style="float: right;"
                   v-model="isMyOJ"
                   active-text="MyOJ"
                   inactive-text="All"
                   @change="statuechange"></el-switch>
        <el-pagination @size-change="handleSizeChange"
                       @current-change="handleCurrentChange"
                       :current-page="currentpage"
                       :page-sizes="[15, 20, 30, 50]"
                       :page-size="pagesize"
                       layout="total, sizes, prev, pager, next, jumper"
                       :total="totalproblem"></el-pagination>

        <el-table :data="tableData"
                  :row-class-name="tableRowClassName"
                  @cell-mouse-enter="changestatistices"
                  @cell-click="problemclick"
                  size="small">
          <el-table-column prop="problem"
                           label="ID"
                           :width="70"></el-table-column>
          <el-table-column prop="title"
                           label="Title"
                           :width="250"></el-table-column>
          <el-table-column prop="level"
                           label="Level"
                           :width="170">
            <template slot-scope="scope1">
              <el-tag id="leveltag"
                      size="medium"
                      :type="problemlevel(scope1.row.level)"
                      disable-transitions
                      hit>{{ scope1.row.level }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="rate"
                           label="A/S"
                           :width="70"></el-table-column>
          <el-table-column prop="tag"
                           label="Tag">
            <template slot-scope="scope">
              <el-tag id="protag"
                      v-for="(name,index) in scope.row.tag"
                      :key="index"
                      size="medium"
                      disable-transitions
                      hit>{{ name }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="score"
                           label="Score"
                           :width="70"></el-table-column>
        </el-table>
        <center>
          <el-pagination @size-change="handleSizeChange"
                         @current-change="handleCurrentChange"
                         :current-page="currentpage"
                         :page-sizes="[15, 20, 30, 50]"
                         :page-size="pagesize"
                         layout="total, sizes, prev, pager, next, jumper"
                         :total="totalproblem"></el-pagination>
        </center>
      </el-card>
    </el-col>

  </el-row>
</template>

<script>
prostatistice;
import prostatistice from "@/components/utils/prostatistice";
import store from "../../store/index";

export default {
  name: "problem",
  components: {
    prostatistice
  },
  data() {
    return {
      currentpage: 1,
      pagesize: 15,
      totalproblem: 10,
      tableData: [],
      tagnames: [],
      ac: 100,
      mle: 100,
      tle: 100,
      rte: 100,
      pe: 100,
      ce: 100,
      wa: 100,
      se: 100,
      title: "Statistics",
      currenttag: "",
      isMyOJ: true,
      searchtext: "",
      searchoj: "MyOJ"
    };
  },
  methods: {
    statuechange(val) {
      if (val === true) {
        this.searchoj = "MyOJ"
      } else {
        this.searchoj = ""
      }
      this.searchtitle()
    },
    searchtitle() {
      this.currentpage = 1;
      this.$axios
        .get(
          "/problemdata/?limit=" +
          this.pagesize +
          "&offset=" +
          (this.currentpage - 1) * this.pagesize +
          "&auth=1&search=" +
          this.searchtext +
          "&oj=" + this.searchoj
        )
        .then(response => {
          for (let i = 0; i < response.data.results.length; i++) {
            if (response.data.results[i]["level"] === "1")
              response.data.results[i]["level"] = "Easy";
            if (response.data.results[i]["level"] === "2")
              response.data.results[i]["level"] = "Medium";
            if (response.data.results[i]["level"] === "3")
              response.data.results[i]["level"] = "Hard";
            if (response.data.results[i]["level"] === "4")
              response.data.results[i]["level"] = "VeryHard";
            if (response.data.results[i]["level"] === "5")
              response.data.results[i]["level"] = "ExtremelyHard";

            response.data.results[i]["rate"] =
              response.data.results[i]["ac"] +
              "/" +
              response.data.results[i]["submission"];

            if (response.data.results[i]["tag"] === null)
              response.data.results[i]["tag"] = ["无"];
            else
              response.data.results[i]["tag"] = response.data.results[i][
                "tag"
                ].split("|");
          }
          this.tableData = response.data.results;
          this.totalproblem = response.data.count;
        });
    },
    tagclick(name) {
      if (this.currenttag.indexOf(name) >= 0) {
        this.$refs[name][0].type = "default";
        let li = this.currenttag.split("+");
        for (let i = 0; i < li.length; i++) {
          if (li[i] === name) {
            li.splice(i, 1);
            break;
          }
        }
        this.currenttag = li.join("+");
      } else {
        this.$refs[name][0].type = "primary";
        let li = this.currenttag.split("+");
        li.push(name);
        this.currenttag = li.join("+");
      }
      this.searchtext = this.currenttag;
      this.currentpage = 1;
      this.$axios
        .get(
          "/problemdata/?limit=" +
          this.pagesize +
          "&offset=" +
          (this.currentpage - 1) * this.pagesize +
          "&auth=1&search=" +
          this.searchtext +
          "&oj=" + this.searchoj
        )
        .then(response => {
          for (let i = 0; i < response.data.results.length; i++) {
            if (response.data.results[i]["level"] === "1")
              response.data.results[i]["level"] = "Easy";
            if (response.data.results[i]["level"] === "2")
              response.data.results[i]["level"] = "Medium";
            if (response.data.results[i]["level"] === "3")
              response.data.results[i]["level"] = "Hard";
            if (response.data.results[i]["level"] === "4")
              response.data.results[i]["level"] = "VeryHard";
            if (response.data.results[i]["level"] === "5")
              response.data.results[i]["level"] = "ExtremelyHard";

            response.data.results[i]["rate"] =
              response.data.results[i]["ac"] +
              "/" +
              response.data.results[i]["submission"];

            if (response.data.results[i]["tag"] === null)
              response.data.results[i]["tag"] = ["无"];
            else
              response.data.results[i]["tag"] = response.data.results[i][
                "tag"
                ].split("|");
          }
          this.tableData = response.data.results;
          this.totalproblem = response.data.count;
        });
    },
    handleSizeChange(val) {
      this.pagesize = val;

      this.$axios
        .get(
          "/problemdata/?limit=" +
          this.pagesize +
          "&offset=" +
          (this.currentpage - 1) * this.pagesize +
          "&auth=1&search=" +
          this.searchtext +
          "&oj=" + this.searchoj
        )
        .then(response => {
          for (let i = 0; i < response.data.results.length; i++) {
            if (response.data.results[i]["level"] === "1")
              response.data.results[i]["level"] = "Easy";
            if (response.data.results[i]["level"] === "2")
              response.data.results[i]["level"] = "Medium";
            if (response.data.results[i]["level"] === "3")
              response.data.results[i]["level"] = "Hard";
            if (response.data.results[i]["level"] === "4")
              response.data.results[i]["level"] = "VeryHard";
            if (response.data.results[i]["level"] === "5")
              response.data.results[i]["level"] = "ExtremelyHard";

            response.data.results[i]["rate"] =
              response.data.results[i]["ac"] +
              "/" +
              response.data.results[i]["submission"];

            if (response.data.results[i]["tag"] === null)
              response.data.results[i]["tag"] = ["无"];
            else
              response.data.results[i]["tag"] = response.data.results[i][
                "tag"
                ].split("|");
          }
          this.tableData = response.data.results;
          this.totalproblem = response.data.count;
        });
    },
    handleCurrentChange(val) {
      this.currentpage = val;
      this.$axios
        .get(
          "/problemdata/?limit=" +
          this.pagesize +
          "&offset=" +
          (this.currentpage - 1) * this.pagesize +
          "&auth=1&search=" +
          this.searchtext +
          "&oj=" + this.searchoj
        )
        .then(response => {
          for (let i = 0; i < response.data.results.length; i++) {
            if (response.data.results[i]["level"] === "1")
              response.data.results[i]["level"] = "Easy";
            if (response.data.results[i]["level"] === "2")
              response.data.results[i]["level"] = "Medium";
            if (response.data.results[i]["level"] === "3")
              response.data.results[i]["level"] = "Hard";
            if (response.data.results[i]["level"] === "4")
              response.data.results[i]["level"] = "VeryHard";
            if (response.data.results[i]["level"] === "5")
              response.data.results[i]["level"] = "ExtremelyHard";

            response.data.results[i]["rate"] =
              response.data.results[i]["ac"] +
              "/" +
              response.data.results[i]["submission"];

            if (response.data.results[i]["tag"] === null)
              response.data.results[i]["tag"] = ["无"];
            else
              response.data.results[i]["tag"] = response.data.results[i][
                "tag"
                ].split("|");
          }
          this.tableData = response.data.results;
          this.totalproblem = response.data.count;
        });
    },
    tableRowClassName({row, rowIndex}) {
      let acpro = this.$store.state.acpro;
      if (acpro)
        if (acpro.indexOf(row.problem + "") !== -1) {
          return "acrow";
        }
      return "";
    },
    problemlevel: function (type) {
      if (type === "Easy") return "info";
      if (type === "Medium") return "success";
      if (type === "Hard") return "";
      if (type === "VeryHard") return "warning";
      if (type === "ExtremelyHard") return "danger";
    },
    changestatistices: function (row, column, cell, event) {
      if (row.submission === 0) {
        this.ac = 0;
        this.mle = 0;
        this.tle = 0;
        this.rte = 0;
        this.pe = 0;
        this.ce = 0;
        this.wa = 0;
        this.se = 0;
      } else {
        this.ac = parseFloat(((row.ac * 100) / row.submission).toFixed(2));
        this.mle = parseFloat(((row.mle * 100) / row.submission).toFixed(2));
        this.tle = parseFloat(((row.tle * 100) / row.submission).toFixed(2));
        this.rte = parseFloat(((row.rte * 100) / row.submission).toFixed(2));
        this.pe = parseFloat(((row.pe * 100) / row.submission).toFixed(2));
        this.ce = parseFloat(((row.ce * 100) / row.submission).toFixed(2));
        this.wa = parseFloat(((row.wa * 100) / row.submission).toFixed(2));
        this.se = parseFloat(((row.se * 100) / row.submission).toFixed(2));
      }
      this.title = row.title;
      this.$refs.prosta.setdata(this.$data);
    },
    problemclick: function (row, column, cell, event) {
      this.$router.push({
        name: "problem_view",
        query: {problemID: row.problem}
      });
    }
  },
  mounted() {
    this.$axios
      .get("/problemdata/?limit=15&offset=0&auth=1&oj=MyOJ")
      .then(response => {
        // console.log(response.data);
        for (let i = 0; i < response.data.results.length; i++) {
          if (response.data.results[i]["level"] === "1")
            response.data.results[i]["level"] = "Easy";
          if (response.data.results[i]["level"] === "2")
            response.data.results[i]["level"] = "Medium";
          if (response.data.results[i]["level"] === "3")
            response.data.results[i]["level"] = "Hard";
          if (response.data.results[i]["level"] === "4")
            response.data.results[i]["level"] = "VeryHard";
          if (response.data.results[i]["level"] === "5")
            response.data.results[i]["level"] = "ExtremelyHard";

          response.data.results[i]["rate"] =
            response.data.results[i]["ac"] +
            "/" +
            response.data.results[i]["submission"];

          if (response.data.results[i]["tag"] === null)
            response.data.results[i]["tag"] = ["无"];
          else
            response.data.results[i]["tag"] = response.data.results[i][
              "tag"
              ].split("|");
        }
        this.tableData = response.data.results;
        this.totalproblem = response.data.count;
      });

    this.$axios.get("/problemtag/").then(response => {
      for (let i = 0; i < response.data.length; i++)
        this.tagnames.push(response.data[i]["tagname"]);
    });
  }
};
</script>


<style scope>
#leveltag {
  text-align: center;
  font-weight: bold;
}

#protag {
  text-align: center;
  font-weight: bold;
  margin-right: 7px;
  margin-bottom: 7px;
}

#tag {
  text-align: center;
  font-weight: bold;
  margin-left: 2px;
  margin-bottom: 5px;
}

.el-row {
  margin-bottom: 20px;
}

.el-table .acrow {
  background: #c7ffb8;
}
</style>
