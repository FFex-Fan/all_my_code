<template>

  <div>
    <section>
      <div class="color"></div>
      <div class="login_box">
        <div class="login_container">
          <div class="login_form">
            <h2>Login</h2>
            <form action="" novalidate>
              <div class="inputBox">
                <input v-model="form.username" type="text" placeholder="Username">
              </div>
              <div class="inputBox">
                <input v-model="form.password" type="password" placeholder="Password">
              </div>
              <div class="inputBox">
                <input type="submit" @click.prevent="loginClick" value="Login">
              </div>
              <p class="forget"> Don't have a account ?
                <RouterLink to="/register">Sign up</RouterLink>
              </p>
            </form>
          </div>
        </div>
      </div>
      <div class="login_box">
        <div class="square" style="--i:0"></div>
        <div class="square" style="--i:1"></div>
        <div class="square" style="--i:2"></div>
        <div class="square" style="--i:3"></div>
        <div class="square" style="--i:4"></div>
        <div class="square" style="--i:5"></div>
      </div>
      <div class="color"></div>
      <div class="color"></div>
    </section>
  </div>
</template>

<script>
import store from "../../store/index";

export default {
  name: "login",
  data() {
    return {
      // dialogLoginVisible: false,
      form: {
        username: "",
        password: ""
      }
    };
  },

  methods: {
    // open() {
    //   this.dialogLoginVisible = true;
    // },
    loginClick() {
      store.dispatch("login", {
        username: this.form.username,
        password: this.$md5(this.form.password),
        $router: this.$router,
        error() { // 将 success() 作为参数传入
          this.$message.error("用户未注册！")
        }
      });

    }
  },
  // beforeRouteLeave(to, from, next) {
  //   next();
  // }
};
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: "黑体";
}

body {
  overflow: hidden;
}

section {
  display: flex;
  justify-content: center;
  align-content: center;
  min-height: 100vh;
  background: linear-gradient(
    to bottom, #f1f4f9, #dff1ff
  );
}

section .color {
  position: absolute;
}

section .color:nth-child(1) {
  top: -150px;
  width: 600px;
  height: 600px;
  background: #ff359b;
}

section .color:nth-child(2) {
  top: -300px;
  left: 100px;
  width: 500px;
  height: 500px;
  float: left;
  background: #fffd87;
}

section .color:nth-child(3) {
  top: -150px;
  right: 100px;
  width: 300px;
  height: 300px;
  float: right;
  background: #00dfff;
}

section .color {
  filter: blur(150px);
}

.login_container {
  margin-top: 35px;
  position: relative;
  width: 400px;
  min-height: 400px;
  background: rgba(255, 255, 255, .1);
  border-radius: 10px;
  display: flex;
  justify-content: center;
  align-content: center;
  backdrop-filter: blur(5px);
  box-shadow: 0 25px 45px rgba(0, 0, 0, .1);
  border: 1px solid rgba(255, 255, 255, .5);
  border-right: 1px solid rgba(255, 255, 255, .2);
  border-bottom: 1px solid rgba(255, 255, 255, .2);
}

.login_form {
  position: relative;
  width: 100%;
  height: 100%;
  padding: 40px;
}

.login_form h2 {
  position: relative;
  color: #fff;
  font-size: 24px;
  font-weight: 600;
  letter-spacing: 1px;
  margin-bottom: 40px;
}

.login_form h2:before {
  content: "";
  position: absolute;
  left: 0;
  bottom: -10px;
  width: 80px;
  height: 4px;
  background: #fff;
}

.login_form .inputBox {
  width: 100%;
  margin-top: 20px;
}

.login_form .inputBox input {
  width: 100%;
  background: rgba(255, 255, 255, .2);
  border: none;
  outline: none;
  padding: 10px 20px;
  border-radius: 35px;
  //box-shadow: 0 25px 45px rgba(0, 0, 0, .5);
  border-right: 1px solid rgba(255, 255, 255, .2);
  border-bottom: 1px solid rgba(255, 255, 255, .2);
  letter-spacing: 1px;
  color: #fff;
  box-shadow: 0 5px 15px rgba(0, 0, 0, .05);
}

.login_form .inputBox input::placeholder {
  color: #fff;
}

.login_form .inputBox input[type="submit"] {
  background-color: #fff;
  color: #666;
  max-width: 100px;
  cursor: pointer;
  margin-bottom: 20px;
  font-weight: 600;
}

.forget {
  margin-top: 5px;
  color: #fff;
}

.forget a {
  color: #fff;
  font-weight: 600;
}

.login_box {
  position: relative;
}

.login_box .square {
  position: absolute;
  width: 100px;
  height: 100px;
  background: rgba(255, 255, 255, .1);
  backdrop-filter: blur(5px);
  box-shadow: 0 25px 45px rgba(0, 0, 0, .1);
  border: 1px solid rgba(255, 255, 255, .5);
  border-right: 1px solid rgba(255, 255, 255, .2);
  border-bottom: 1px solid rgba(255, 255, 255, .2);
  border-radius: 10px;
}

.login_box .square:nth-child(1) {
  top: -50px;
  right: -60px;
  width: 100px;
  height: 100px;
}

.login_box .square:nth-child(1) {
  top: 280px;
  right: 370px;
  width: 200px;
  height: 200px;
}

.login_box .square:nth-child(2) {
  top: 20px;
  right: 30px;
  width: 70px;
  height: 70px;
}

.login_box .square:nth-child(3) {
  top: 110px;
  right: -100px;
  width: 40px;
  height: 40px;
}

.login_box .square:nth-child(4) {
  top: 310px;
  right: -110px;
  width: 150px;
  height: 150px;
}

.login_box .square:nth-child(5) {
  top: 400px;
  right: 140px;
  width: 60px;
  height: 60px;
}

.login_box .square:nth-child(6) {
  top: 90px;
  right: 390px;
  width: 60px;
  height: 60px;
}

.login_box .square {
  animation: animate 10s linear infinite;
  animation-delay: calc(-1s * var(--i));
}

@keyframes animate {
  0%, 100% {
    transform: translateY(-40px);
  }
  50% {
    transform: translateY(40px);
  }
}

</style>
