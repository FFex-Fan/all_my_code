import Vue from 'vue'
import Router from 'vue-router'
import VueRouter from "vue-router";
import store from "../store";


const homepage = r => require.ensure([], () => r(require("@/components/main")), 'main"');
const problem = r => require.ensure([], () => r(require("@/components/mainpage/problem")), 'mainpage');
const statue = r => require.ensure([], () => r(require("@/components/mainpage/statue")), 'mainpage');
const user = r => require.ensure([], () => r(require("@/components/mainpage/user")), 'mainpage');
const login = r => require.ensure([], () => r(require("@/components/base_page/login")), 'base_page');
const register = r => require.ensure([], () => r(require("@/components/base_page/register")), 'base_page');
const setting = r => require.ensure([], () => r(require("@/components/mainpage/setting")), 'mainpage');
const contest = r => require.ensure([], () => r(require("@/components/mainpage/contest")), 'mainpage');
const contestdetail = r => require.ensure([], () => r(require("@/components/contest/contestdetail")), 'contest');
const problemdetail = r => require.ensure([], () => r(require("@/components/problem/problemdetail")), 'problem');
const problem_view = r => require.ensure([], () => r(require("@/components/problem/problemView")), 'problem');
const rank = r => require.ensure([], () => r(require("@/components/mainpage/rank")), 'mainpage');
const admin = r => require.ensure([], () => r(require("@/components/mainpage/admin")), 'mainpage');
const billboard = r => require.ensure([], () => r(require("@/components/mainpage/billboard")), 'mainpage');
const blog = r => require.ensure([], () => r(require("@/components/mainpage/blog")), 'mainpage');
const homework = r => require.ensure([], () => r(require("@/components/mainpage/homework")), 'mainpage');
const givechoiceproblemscore = r => require.ensure([], () => r(require("@/components/admin/givechoiceproblemscore")), 'admin');


Vue.use(Router)

// 获取原型对象push函数
const originalPush = VueRouter.prototype.push

// 获取原型对象replace函数
const originalReplace = VueRouter.prototype.replace

// 修改原型对象中的push函数
VueRouter.prototype.push = function push(location){
  return originalPush.call(this , location).catch(err=>err)
}

// 修改原型对象中的replace函数
VueRouter.prototype.replace = function replace(location){
  return originalReplace.call(this , location).catch(err=>err)
}


export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      name: 'homepage',
      component: homepage
    },
    {
      path:'/login',
      name: 'login',
      component: login
    },
    {
      path:'/register',
      name: 'register',
      component: register
    },
    {
      path: '/problem',
      name: 'problem',
      component: problem,
      // meta: {isAuth: true, title: '问题'},
      // beforeEnter:(to, from, next) => {
      //   if (store.state.is_login) {
      //     next();
      //     location.reload();
      //   }
      //   next();
      // }
    },
    {
      path: '/problemdetail',
      name: 'problemdetail',
      component: problemdetail,
    },
    {
      path: '/problem_view',
      name: 'problem_view',
      component: problem_view,
    },
    {
      path: '/admin',
      name: 'admin',
      component: admin
    },
    {
      path: '/statue',
      name: 'statue',
      component: statue
    },
    {
      path: '/user',
      name: 'user',
      component: user
    },
    {
      path: '/setting',
      name: 'setting',
      component: setting
    },
    {
      path: '/contest',
      name: 'contest',
      component: contest
    },
    {
      path: '/contest/:contestID',
      name: 'contestdetail',
      component: contestdetail,
    },
    {
      path: '/rank',
      name: 'rank',
      component: rank,
    },
    {
      path: '/billboard',
      name: 'billboard',
      component: billboard,
    },
    {
      path: '/blog',
      name: 'blog',
      component: blog,
    },
    {
      path: '/homework',
      name: 'homework',
      component: homework,
    },
    {
      path: '/givechoiceproblemscore',
      name: 'givechoiceproblemscore',
      component: givechoiceproblemscore,
    }
  ]
})
