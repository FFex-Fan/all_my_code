<template>
  <div class="box">
    <div class="box_left">
      <el-tabs v-model="activeName" @tab-click="handleClick">
        <el-tab-pane name="first">
          <span slot="label">
            <i class="bi bi-code-slash" style="margin-left: 20px;margin-right: 8px;"></i>题目描述
          </span>
          <el-row>
            <el-card shadow="always">
              <!-- Title -->
              <el-row :gutter="18" id="title">{{ title }}</el-row>
              <br>
              <!-- Direction -->
              <el-row :gutter="18"
                      id="des">Description
              </el-row>
              <el-row :gutter="18" class="detail">
                <div style="margin-right:50px;word-break:break-all;white-space:pre-line;" v-html="des" :key="des"></div>
              </el-row>

              <img :src="'data:image/jpeg;base64,'+imgcode"
                   class="img-responsive" v-if="imgcode!=''" alt="">
              <br>
              <!-- Sample -->
              <el-row :gutter="18" style="left:10px">
                <el-row :gutter="18" v-for="(item,index) in sinput.length" :key="index">
                  <!-- Sample Input-->
                  <el-col :span="10" id="text">
                    <el-row :gutter="18" id="des">Sample Input {{ item }}</el-row>
                    <el-row :gutter="18" id="data">{{ sinput[index] }}</el-row>
                  </el-col>

                  <!-- Sample Output-->
                  <el-col :span="10" id="text">
                    <el-row :gutter="18" id="des">Sample Output {{ item }}</el-row>
                    <el-row :gutter="18" id="data">{{ soutput[index] }}</el-row>
                  </el-col>
                </el-row>
                <br>
              </el-row>

              <!-- Hint  -->
              <el-row :gutter="18" id="des">Hint</el-row>
              <el-row :gutter="18" class="detail">
                <div style="margin-right:50px;word-break:break-all;white-space:pre-line;" v-html="hint"></div>
              </el-row>
              <br>
            </el-card>
          </el-row>
        </el-tab-pane>

        <el-tab-pane name="second">
          <span slot="label"><i class="bi bi-card-text" style="margin-right: 8px;"></i>提交记录</span>
          配置管理
        </el-tab-pane>
      </el-tabs>
    </div>

    <div class="resize"></div>

    <div class="box_right">
      <el-row style="margin-top: 20px; margin-left: 20px;">
        <el-col :span="4">
          <div style="padding: 6px 0 5px 10px; font-size: 14px">Language:</div>
        </el-col>
        <el-col :span="16">
          <el-select size="mini" v-model="language" placeholder="请选择"
                     style="margin-bottom: 20px; margin-top: 3px;" @change="changetemplate">
            <languageselect></languageselect>
          </el-select>
        </el-col>

        <el-button size="small"
                   style="float:right; margin-right: 20px"
                   icon="el-icon-refresh-right" circle
                   @click="code = ''"></el-button>

      </el-row>

      <div id="code_area">
        <el-row>
          <codemirror v-model="code" :options="cmOptions"
                      ref="myCm" class="code"
          ></codemirror>
        </el-row>
      </div>

      <div>
        <el-row style="height: 80px">
          <el-col :span="20">
            <el-button plain :type="judgetype"
                       v-if="submitbuttontext !== ''"
                       :loading="loadingshow"
                       style="margin-left:10px;margin-top:10px;"
                       @click="showdialog">{{ submitbuttontext }}
            </el-button>
          </el-col>

          <el-button round plain type="success" size="small"
                     @click="submit"
                     style=" float:right; font-weight:bold; margin-top:10px; margin-right: 30px "
          >Submit
          </el-button>


        </el-row>
      </div>
    </div>
  </div>
</template>


<script>
import moment from "moment";
import {codemirror} from "vue-codemirror";
import statusmini from "@/components/utils/statusmini";
import languageselect from "@/components/utils/languageselect";
import prostatistice from "@/components/utils/prostatistice";


import 'codemirror/addon/edit/matchbrackets'
import 'codemirror/addon/edit/closebrackets'
import Clipboard from 'clipboard'

require("codemirror/lib/codemirror.css");
require("codemirror/theme/dracula.css");
require("codemirror/theme/eclipse.css");
require("codemirror/theme/solarized.css");
require("codemirror/theme/monokai.css");
require("codemirror/mode/clike/clike");
require("codemirror/mode/htmlmixed/htmlmixed");
require("codemirror/addon/edit/matchbrackets")
require("codemirror/addon/hint/show-hint");
require("codemirror/addon/hint/show-hint.css");
require("codemirror/addon/hint/javascript-hint");
require("codemirror/addon/hint/anyword-hint");

export default {
  name: "problemdetail",
  mounted() {
    this.dragControllerDiv();
    let that = this;
    that.clientHeight = `${document.documentElement.clientHeight}`;//获取浏览器可视区域高度
    that.editor = this.$refs.myCm.codemirror; // 获取codemirror对象
    that.editor.setSize('auto', this.clientHeight - 250); // 设置codemirror高度
    window.onresize = function () { // 监听屏幕
      that.clientHeight = `${document.documentElement.clientHeight}`;
      that.editor.setSize('auto', parseFloat(that.clientHeight) - 250); // 设置代码区域高度
    }
  },
  components: {
    codemirror,
    statusmini,
    prostatistice,
    languageselect
  },
  data() {
    return {
      activeName: 'first',
      imgcode: "",
      ip: "",
      userip: "",
      cmOptions: {
        value: '', // 初始内容
        mode: "text/x-c++src",
        line: true,
        theme: "monokai", // 主题
        tabSize: 4, // 制表符的宽度
        indentUnit: 4, // 一个块应该缩进多少个空格（无论这在编辑语言中意味着什么）。默认值为 2。
        firstLineNumber: 1, // 从哪个数字开始计算行数。默认值为 1。
        readOnly: false, // 只读
        autoCloseBrackets: true, // 自动闭合符号
        matchBrackets: true, // 在光标点击紧挨{、]括号左、右侧时，自动突出显示匹配的括号 }、】
        // autorefresh: true,
        smartIndent: true, // 上下文缩进
        electricChars: true, // 换行重新调整缩进
        dragDrop: true,
        lineNumbers: true, // 是否显示行号
        styleActiveLine: true, // 高亮选中行
        viewportMargin: Infinity, //处理高度自适应时搭配使用
        showCursorWhenSelecting: true, // 当选择处于活动状态时是否应绘制游标
      },
      title: "",
      des: "",
      input: "",
      output: "",
      sinput: ["", ""],
      soutput: ["", ""],
      author: "",
      addtime: "",
      oj: "",
      proid: "",
      source: "",
      time: "",
      memory: "",
      hint: "",
      tagnames: [],
      activeNames: ["1", "2", "3", "4", "5", "6"],
      level: "Easy",
      code: "",
      language: "C++",

      codetemplate: {},

      ac: 100,
      mle: 100,
      tle: 100,
      rte: 100,
      pe: 100,
      ce: 100,
      wa: 100,
      se: 100,
      submitbuttontext: '',
      judgetype: "primary",
      loadingshow: false,
      submitid: -1
    };
  },
  watch: {
    des: function () {
      console.log('data changed');
      this.$nextTick().then(() => {
        this.reRender();
      });
    }
  },
  created() {
    let myip = require('ip');
    this.userip = myip.address();
    this.ID = this.$route.query.problemID;
    if (!this.ID) {
      this.$message.error("参数错误" + "(" + this.ID + ")");
      return;
    }
    let auth = 1;
    this.$axios
      .get("/showpic", {
        params: {
          ProblemId: this.$route.query.problemID
        }
      })
      .then(res => {
        this.imgcode = res.data;
      }).catch(error => {
      this.imgcode = ''
    });
    this.$axios
      .get("/problem/" + this.ID + "/")
      .then(response => {
        auth = response.data.auth;
        if ((auth == 2 || auth == 3) && (sessionStorage.type == 1 || sessionStorage.type == "")) {
          this.title = "非法访问！请在比赛中访问题目！";
          this.$message.error("服务器错误！(无权限)");
          return;
        }
        this.proid = this.ID
        this.des = response.data.des;
        this.input = response.data.input;
        this.output = response.data.output;
        this.sinput = response.data.sinput.split("|#)"); //分隔符
        this.soutput = response.data.soutput.split("|#)");
        this.author = response.data.author;
        this.addtime = response.data["addtime"] = moment(
          response.data["addtime"]
        ).format("YYYY-MM-DD HH:mm:ss");

        this.oj = response.data.oj;
        this.source = response.data.source;
        this.time = response.data.time + "MS";
        this.memory = response.data.memory + "MB";
        this.hint = response.data.hint;

        let li = response.data.template.split("*****")
        for (let i = 1; i < li.length; i += 2) {
          this.codetemplate[li[i]] = li[i + 1]
        }
        this.code = this.codetemplate[this.language]

        if (this.oj != "MyOJ") {
          this.proid = this.source
        }


        this.$axios
          .get("/problemdata/" + this.ID + "/")
          .then(response => {
            if (response.data["level"] == "1") response.data["level"] = "Easy";
            if (response.data["level"] == "2")
              response.data["level"] = "Medium";
            if (response.data["level"] == "3") response.data["level"] = "Hard";
            if (response.data["level"] == "4")
              response.data["level"] = "VeryHard";
            if (response.data["level"] == "5")
              response.data["level"] = "ExtremelyHard";

            if (response.data["tag"] == null) response.data["tag"] = ["无"];
            else response.data["tag"] = response.data["tag"].split("|");

            if (response.data.submission == 0) {
              this.ac = 0;
              this.mle = 0;
              this.tle = 0;
              this.rte = 0;
              this.pe = 0;
              this.ce = 0;
              this.wa = 0;
              this.se = 0;
            } else {
              this.ac = parseFloat(
                ((response.data.ac * 100) / response.data.submission).toFixed(2)
              );
              this.mle = parseFloat(
                ((response.data.mle * 100) / response.data.submission).toFixed(
                  2
                )
              );
              this.tle = parseFloat(
                ((response.data.tle * 100) / response.data.submission).toFixed(
                  2
                )
              );
              this.rte = parseFloat(
                ((response.data.rte * 100) / response.data.submission).toFixed(
                  2
                )
              );
              this.pe = parseFloat(
                ((response.data.pe * 100) / response.data.submission).toFixed(2)
              );
              this.ce = parseFloat(
                ((response.data.ce * 100) / response.data.submission).toFixed(2)
              );
              this.wa = parseFloat(
                ((response.data.wa * 100) / response.data.submission).toFixed(2)
              );
              this.se = parseFloat(
                ((response.data.se * 100) / response.data.submission).toFixed(2)
              );
            }
            this.title = response.data.title;
            this.level = response.data.level;
            this.tagnames = response.data.tag;
            this.$refs.prosta.setdata(this.$data)
            console.log("asaaaaa ");
            // console.log(this.$refs["Statusmini"])
            this.$refs["Statusmini"].setstatus(this.ID, sessionStorage.username, "");


          })
          .catch(error => {
            // this.$message.error("服务器错误111111！" + "(" + JSON.stringify(error.response) + ")");
          });
      })
      .catch(error => {
        this.title = "非法访问！请在比赛中访问题目！";
        this.$message.error("服务器错误！" + "(" + JSON.stringify(error.response) + ")");
      });
  },
  methods: {
    handleClick(tab, event) {
      console.log(tab, event);
    },
    showdialog() {
      if (this.submitid != -1)
        this.$refs["Statusmini"].showdialog(this.submitid)
    },
    changetemplate(lang) {
      let t = this.codetemplate[lang]
      if (t) {
        this.$confirm("确定切换语言吗？", "切换后当前代码不会保存！", {
          confirmButtonText: "确定",
          cancelButtonText: "取消",
          type: "warning"
        }).then(() => {

          this.code = this.codetemplate[lang]
        })
      }


    },
    reRender() {
      if (window.MathJax) {
        console.log('rendering mathjax');
        MathJax.Hub.Config({
          tex2jax: {
            inlineMath: [['$', '$'], ["\\(", "\\)"]],
            displayMath: [['$$', '$$'], ["\\[", "\\]"]]
          }
        });
        window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub], () => console.log('done'));
      }
    },
    onCopy(e) {
      this.$message.success("复制成功！");
    },
    onError(e) { // 复制失败
      this.$message.error("复制失败：" + e);
    },
    problemlevel: function (type) {
      if (type == "Easy") return "info";
      if (type == "Medium") return "success";
      if (type == "Hard") return "";
      if (type == "VeryHard") return "warning";
      if (type == "ExtremelyHard") return "danger";
    },
    submit: function () {
      if (this.addtime == "") {
        this.$message.error("非法操作！");
        return;
      }
      if (!sessionStorage.username) {
        this.$message.error("请先登录！");
        return;
      }
      if (!this.code) {
        this.$message.error("请输入代码！");
        return;
      }
      if (!this.language) {
        this.$message.error("请选择语言！");
        return;
      }

      if (this.code.length < 20) {
        this.$message.error("代码过短！");
        return;
      }

      this.$confirm("确定提交吗？", "提交", {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning"
      }).then(() => {
        this.$message({
          type: "success",
          message: "提交中..."
        });
        this.$axios.get("/currenttime/").then(response2 => {
          // console.log(this.userip);
          let curtime = response2.data;
          //this.$axios.get("/")
          this.$axios
            .post("/judgestatusput/", {
              user: sessionStorage.username,
              oj: this.oj,
              problem: this.ID,
              result: -1,
              time: 0,
              memory: 0,
              length: this.code.length,
              language: this.language,
              submittime: curtime,
              judger: "waiting for judger",
              contest: 0,
              code: this.code,
              testcase: 0,
              message: this.oj == "MyOJ" ? "0" : (this.proid + ""),
              problemtitle: (this.oj == "MyOJ" ? "MyOJ" : "") + (this.oj == "MyOJ" ? ' - ' : "") + (this.oj == "MyOJ" ? this.proid : "") + ' ' + this.title,
              rating: parseInt(sessionStorage.rating),
              ip: this.userip
            })
            .then(response => {
              this.$message({
                message: "提交成功！",
                type: "success"
              });
              clearInterval(this.$store.state.submittimer);
              this.submitid = response.data.id;
              this.submitbuttontext = "Pending";
              this.judgetype = "info";
              this.loadingshow = true;
              //创建一个全局定时器，定时刷新状态
              this.$store.state.submittimer = setInterval(this.timer, 3000);
            })
            .catch(error => {
              this.$message.error("服务器错误！" + "(请检查编码（代码需要utf-8编码）或联系管理员)");
            });
        });
      });
    },
    timer: function () {
      if (this.submitbuttontext == "提交后请勿重复刷新/支持将文件拖入代码框") return;
      this.$axios.get("/judgestatus/" + this.submitid + "/").then(response => {
        this.loadingshow = false;
        let testcase = response.data["testcase"];
        if (response.data["result"] == "-1") {
          response.data["result"] = "Pending";
          this.loadingshow = true;
          this.judgetype = "info";
        }

        if (response.data["result"] == "-2") {
          response.data["result"] = "Judging";
          this.loadingshow = true;
          this.judgetype = "";
        }

        if (response.data["result"] == "-3") {
          response.data["result"] = "Wrong Answer on test " + testcase;
          this.judgetype = "danger";
          clearInterval(this.$store.state.submittimer);
          if (testcase == "?")
            response.data["result"] = "Wrong Answer"
        }

        if (response.data["result"] == "-4") {
          response.data["result"] = "Compile Error";
          this.judgetype = "warning";
          clearInterval(this.$store.state.submittimer);
        }

        if (response.data["result"] == "-5") {
          response.data["result"] = "Presentation Error on test " + testcase;
          this.judgetype = "warning";
          clearInterval(this.$store.state.submittimer);
          if (testcase == "?")
            response.data["result"] = "Presentation Error"
        }

        if (response.data["result"] == "-6") {
          response.data["result"] = "Waiting";
          this.loadingshow = true;
          this.judgetype = "info";
        }

        if (response.data["result"] == "0") {
          response.data["result"] = "Accepted";
          this.judgetype = "success";
          clearInterval(this.$store.state.submittimer);
        }

        if (response.data["result"] == "1") {
          response.data["result"] = "Time Limit Exceeded on test " + testcase;
          this.judgetype = "warning";
          clearInterval(this.$store.state.submittimer);
          if (testcase == "?")
            response.data["result"] = "Time Limit Exceeded"
        }

        if (response.data["result"] == "2") {
          response.data["result"] = "Time Limit Exceeded on test " + testcase;
          this.judgetype = "warning";
          clearInterval(this.$store.state.submittimer);
          if (testcase == "?")
            response.data["result"] = "Time Limit Exceeded"
        }

        if (response.data["result"] == "3") {
          response.data["result"] = "Memory Limit Exceeded on test " + testcase;
          this.judgetype = "warning";
          clearInterval(this.$store.state.submittimer);
          if (testcase == "?")
            response.data["result"] = "Memory Limit Exceeded"
        }

        if (response.data["result"] == "4") {
          response.data["result"] = "Runtime Error on test " + testcase;
          this.judgetype = "warning";
          clearInterval(this.$store.state.submittimer);
          if (testcase == "?")
            response.data["result"] = "Runtime Error"
        }

        if (response.data["result"] == "5") {
          response.data["result"] = "System Error";
          this.judgetype = "danger";
          clearInterval(this.$store.state.submittimer);
        }

        this.submitbuttontext = response.data["result"];
        this.$refs["Statusmini"].reflash()
      });
    },
    dragControllerDiv: function () {
      let resize = document.getElementsByClassName('resize');
      let left = document.getElementsByClassName('box_left');
      let mid = document.getElementsByClassName('box_right');
      let box = document.getElementsByClassName('box');
      for (let i = 0; i < resize.length; i++) {
        // 鼠标按下事件
        resize[i].onmousedown = function (e) {
          //颜色改变提醒
          resize[i].style.background = '#818181';
          let startX = e.clientX;
          resize[i].left = resize[i].offsetLeft;
          // 鼠标拖动事件
          document.onmousemove = function (e) {
            let endX = e.clientX;
            let moveLen = resize[i].left + (endX - startX); // （endx-startx）=移动的距离。resize[i].left+移动的距离=左边区域最后的宽度
            let maxT = box[i].clientWidth - resize[i].offsetWidth; // 容器宽度 - 左边区域的宽度 = 右边区域的宽度

            if (moveLen < 32) moveLen = 32; // 左边区域的最小宽度为32px
            if (moveLen > maxT - 150) moveLen = maxT - 150; //右边区域最小宽度为150px

            resize[i].style.left = moveLen; // 设置左侧区域的宽度

            for (let j = 0; j < left.length; j++) {
              left[j].style.width = moveLen + 'px';
              mid[j].style.width = (box[i].clientWidth - moveLen - 10) + 'px';
            }
          };
          // 鼠标松开事件
          document.onmouseup = function (evt) {
            //颜色恢复
            resize[i].style.background = '#d6d6d6';
            document.onmousemove = null;
            document.onmouseup = null;
            resize[i].releaseCapture && resize[i].releaseCapture(); //当你不在需要继续获得鼠标消息就要应该调用ReleaseCapture()释放掉
          };
          resize[i].setCapture && resize[i].setCapture(); //该函数在属于当前线程的指定窗口里设置鼠标捕获
          return false;
        };
      }
    },
  },
  destroyed() {
    clearInterval(this.$store.state.submittimer);
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
#title {
  color: Black;
  font-weight: bold;
  left: 10px;
  font-size: 22px;
}

#des {
  color: black;
  font-weight: bold;
  left: 15px;
  font-size: 16px;
}

#text {
  font-weight: normal;
  font-size: 15px;
  white-space: pre-wrap;
  margin-right: 40px;
}

#data {
  left: 20px;
  padding: 5px 10px;
  color: dimgray;
  background: #f8f8f9;
  border: 1px dashed #e9eaec;
  margin-bottom: 15px;
}

.box {
  width: 100%;
  height: 100%;
  display: flex;
  margin: 1% 0;
  overflow: hidden;
  box-shadow: -1px 9px 10px 3px rgba(0, 0, 0, 0.11);
}

.box_left {
  float: left;
  border-radius: 20px;
  margin-left: 20px;
  width: calc(45% - 5px); /*左侧初始化宽度*/
  height: 100%;
  background: #FFFFFF;
  //padding: 10px 0;
}

/*拖拽区div样式*/
.resize {
  cursor: col-resize;
  float: left;
  position: relative;
  top: 45%;
  background-color: #d6d6d6;
  border-radius: 5px;
  margin-top: 5px;
  width: 5px;
  height: 77vh;
  background-size: cover;
  background-position: center;
  /*z-index: 99999;*/
  font-size: 32px;
  color: white;
}

/*拖拽区鼠标悬停样式*/
.resize:hover {
  color: lightblue;
}

.box_right {
  //float: right;
  border-radius: 20px;
  width: 55%; /*右侧初始化宽度*/
  flex-wrap: nowrap;
  height: 100%;
  background: #fff;
  box-shadow: -1px 4px 5px 3px rgba(0, 0, 0, 0.11);
}

el-tab-pane:nth-child(1) {
  cursor: none;
}

.img-responsive {
  display: inline-block;
  height: auto;
  max-width: 75%;
}

#code_area {
  height: 70vh;
}

.code {
  font-size: 16px;
}

.detail {
  left: 30px;
  font-size: 16px;
  margin-top: 10px;
}
</style>

