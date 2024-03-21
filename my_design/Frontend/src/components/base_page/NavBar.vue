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
                     v-if="!this.$store.state.user.is_login"
          >Register</el-button>
        </RouterLink>

        <RouterLink to="/login/">
          <el-button round
                     id="button"
                     size="middle"
                     v-if="!this.$store.state.user.is_login"
          >Login</el-button>
        </RouterLink>

        <el-dropdown id="user_login_show"
                     v-if="this.$store.state.user.is_login"
                     @command="handleCommand"
                     :show-timeout="100">
          <span class="el-dropdown-link">
            {{ this.$store.state.user.username }}
            <i class="el-icon-arrow-down el-icon--right"></i>
          </span>
          <el-dropdown-menu slot="dropdown">
            <el-dropdown-item command="home">Home</el-dropdown-item>
            <el-dropdown-item command="submission">Submission</el-dropdown-item>
            <el-dropdown-item command="setting">Setting</el-dropdown-item>
            <el-dropdown-item command="admin"
                              divided
                              v-if="is_admin"
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
      this.$store.dispatch("handleCommand", {
        command: command,
        $router: this.$router,
      })
    },
  },

  components: {
    "login": resolve => require(['@/components/base_page/login'], resolve),
    "register": resolve => require(['@/components/base_page/register'], resolve)
  },

  data() {
    return {
      activeIndex: "1",
      school: "MyOJ",
      username: sessionStorage.username,
      is_admin: false
    };
  },
  mounted() {
    this.is_admin = parseInt(sessionStorage.type) === 2 || parseInt(sessionStorage.type) === 3;

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

#user_login_show {
  float: right;
  margin-top: 20px;
  margin-right: 50px;
  font-size: 18px;
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

.el-icon-arrow-down {
  font-size: 12px;
}

</style>
