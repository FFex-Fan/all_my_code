<template>
  <div id="app" class="container" style="top:0;left:0;">
    <div class="head">
      <el-menu :default-active="$route.path"
               mode="horizontal"
               v-bind:router="true"
               class="el-menu-demo"
               id="nav">
        <el-menu-item index="/"
                      id="title">{{ school }}
        </el-menu-item>
        <el-menu-item index="/">
          <i class="bi bi-balloon" style="margin-right: 5px"></i>Home
        </el-menu-item>
        <el-menu-item index="/problem">
          <i class="el-icon-menu" style="margin-bottom: 2px"></i>Problem
        </el-menu-item>
        <el-menu-item index="/statue">
          <i class="bi bi-activity" style="margin-right: 5px"></i>Status
        </el-menu-item>
        <el-menu-item index="/contest">
          <i class="el-icon-s-data" style="margin-bottom: 2px"></i>Contest
        </el-menu-item>
        <!--        <el-menu-item index="/rank">-->
        <!--          <i class="el-icon-star-on"></i>Rank-->
        <!--        </el-menu-item>-->

        <RouterLink to="/register">
          <el-button round
                     id="button"
                     @click="registeropen"
                     v-show="!loginshow">Register
          </el-button>
        </RouterLink>

        <RouterLink to="/login">
          <el-button round
                     id="button"
                     @click="loginopen"
                     v-show="!loginshow">Login
          </el-button>
        </RouterLink>

        <el-dropdown id="user"
                     v-show="loginshow"
                     @command="handleCommand"
                     :show-timeout="100"
                     :split-button="true">
          <span class="el-dropdown-link">Welcome {{ username }}</span>
          <el-dropdown-menu slot="dropdown">
            <el-dropdown-item command="home">Home</el-dropdown-item>
            <el-dropdown-item command="submission">Submission</el-dropdown-item>
            <el-dropdown-item command="setting">Setting</el-dropdown-item>
            <el-dropdown-item command="admin"
                              divided
                              v-show="isadmin">Admin
            </el-dropdown-item>
            <el-dropdown-item command="logout"
                              divided>Logout
            </el-dropdown-item>
          </el-dropdown-menu>
        </el-dropdown>
      </el-menu>

<!--      <register ref="registerdialog"></register>-->

<!--      <login ref="logindialog"></login>-->
    </div>


    <!--    <el-backtop :bottom="50">-->
    <!--      <div style="{-->
    <!--        height: 100%;-->
    <!--        width: 100%;-->
    <!--        background-color: #f2f5f6;-->
    <!--        /*box-shadow: 0 0 6px rgba(0,0,0, .12);*/-->
    <!--        box-shadow: 0 0 6px rgb(0,0,0);-->
    <!--        text-align: center;-->
    <!--        line-height: 40px;-->
    <!--        color: #1989fa;-->
    <!--      }">UP-->
    <!--      </div>-->
    <!--    </el-backtop>-->

    <div class="content">
      <transition name="el-fade-in-linear"
                  mode="out-in">
        <router-view id="route"></router-view>
      </transition>
    </div>

    <div class="footer">
    </div>
  </div>
</template>

<script>
export default {
  name: "App",
  components: {
    "login": resolve => require(['@/login'], resolve),
    "register": resolve => require(['@/register'], resolve)
  },

  data() {
    return {
      activeIndex: "1",
      school: "OJ",
      loginshow: sessionStorage.username,
      username: sessionStorage.username,
      // name: sessionStorage.name,
      isadmin: false
    };
  },
  mounted() {
    this.isadmin = sessionStorage.type === 2 || sessionStorage.type === 3;

    var sb = this.$store.state.sb
    if (sb === undefined) {
      this.$axios
        .get("/settingboard/")
        .then(res => {
          if (res.data.length > 0) this.school = res.data[0].ojname;
          else this.school = "MyOJ";
          this.$store.state.sb = res.data
        });
    } else {
      if (sb.length > 0) this.school = sb[0].ojname;
      else this.school = "MyOJ";
    }


  },
  methods: {
    loginopen() {
      // this.$refs.logindialog.open();
    },
    registeropen() {
      // this.$refs.registerdialog.open();
    },

    handleCommand(command) {
      if (command === "logout") {
        this.$axios
          .get("/logout/")
          .then(response => {
            this.$message({
              message: "登出成功！",
              type: "success"
            });
            sessionStorage.setItem("username", "");
            sessionStorage.setItem("name", "");
            sessionStorage.setItem("rating", "");
            sessionStorage.setItem("type", "");
            this.loginshow = 0;
            this.username = "";
            this.$router.go(0);
          })
          .catch(error => {
            this.$message.error(
              "服务器错误！" + "(" + JSON.stringify(error.response.data) + ")"
            );
          });
      }
      if (command === "home") {
        this.$router.push({
          name: "user",
          query: {username: sessionStorage.username}
        });
      }
      if (command === "setting") {
        this.$router.push({
          name: "setting",
          params: {username: sessionStorage.username}
        });
      }
      if (command === "submission") {
        this.$router.push({
          name: "statue",
          query: {username: sessionStorage.username}
        });
      }
      if (command === "admin") {
        this.$router.push({
          name: "admin"
        });
      }
      if (command === "classes") {
        this.$router.push({
          name: "classes"
        });
      }
    }
  }
};
</script>

<style scoped>
.el-dropdown-link {
  cursor: pointer;
  color: #409eff;
}

#button {
  float: right;
  margin: 10px;
}

#user {
  float: right;
  margin: 10px;
}


#nav {
  background-color: #ffffff;
  position: relative;
  left: 0;
  top: 0;
  z-index: 5;
  width: 100%;
}

#route {
  position: relative;
  top: 10px;
}

#title {
  font-size: 20px;
  font-weight: bold;
}

.content {
  margin-top: 15px;
}

.el-row {
  margin-bottom: 20px;
}

.head {
  //height: 70px;
}

.footer {
  margin-top: 15px;
  margin-bottom: 10px;
  text-align: center;
  font-size: small;
}
</style>
