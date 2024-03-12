<template>
  <div class="head">
    <el-menu :default-active="$route.path"
             mode="horizontal"
             v-bind:router="true"
             class="el-menu-demo"
             id="nav">
      <el-menu-item id="title" index="/"
      >{{ school }}
      </el-menu-item>
      <el-menu-item index="/">
        <i class="el-icon-s-home" style="margin-bottom: 2px"></i>Home
      </el-menu-item>
      <el-menu-item index="/problem/">
        <i class="el-icon-menu" style="margin-bottom: 2px"></i>Problem
      </el-menu-item>
      <el-menu-item index="/statue/">
        <i class="bi bi-activity" style="margin-right: 5px"></i>Status
      </el-menu-item>
      <el-menu-item index="/contest/">
        <i class="el-icon-s-data" style="margin-bottom: 2px"></i>Contest
      </el-menu-item>
      <!--        <el-menu-item index="/rank">-->
      <!--          <i class="el-icon-star-on"></i>Rank-->
      <!--        </el-menu-item>-->

      <div class="nav_right">
        <RouterLink to="/register/">
          <el-button round
                     id="button"
                     size="middle"
                     v-show="!loginshow">Register
          </el-button>
        </RouterLink>

        <RouterLink to="/login/">
          <el-button round
                     id="button"
                     size="middle"
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
                              v-show="isadmin"
            >Admin
            </el-dropdown-item>
            <el-dropdown-item command="logout"
                              divided>Logout
            </el-dropdown-item>
          </el-dropdown-menu>
        </el-dropdown>

      </div>
    </el-menu>
  </div>
</template>

<script>
export default {
  name: 'NavBar',
  methods: {
    handleCommand(command) {
      if (command === "logout") {
        this.$axios
          .get("/logout/")
          .then(response => {
            this.$message({
              message: "登出" +
                "成功！",
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

  },

  components: {
    "login": resolve => require(['@/components/base_page/login'], resolve),
    "register": resolve => require(['@/components/base_page/register'], resolve)
  },

  data() {
    return {
      activeIndex: "1",
      school: "MyOJ",
      loginshow: sessionStorage.username,
      username: sessionStorage.username,
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
}
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
  margin-top: 8px;
  margin-bottom: 10px;
}


#nav {
  background-color: #ffffff;
  position: relative;
  left: 0;
  top: 0;
  z-index: 5;
  width: 100%;
}

#title {
  font-size: 28px;
  font-weight: bold;
  margin-left: 60px;
  font-family: "Chalkduster", "Brush Script MT", serif;
  width: 15%;
  text-align: left;
  border-bottom: none;
}


.nav_right {
  margin-right: 30px;
}

.head {
  font-size: 16px;
}

</style>
