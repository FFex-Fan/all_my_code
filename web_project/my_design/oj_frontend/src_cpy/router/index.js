import Vue from 'vue'
import Router from 'vue-router'

const homepage = r => require.ensure([], () => r(require("@/components/main")), 'main"');
const problem = r => require.ensure([], () => r(require("@/components/mainpage/problem")), 'mainpage');
const statue = r => require.ensure([], () => r(require("@/components/mainpage/statue")), 'mainpage');
const user = r => require.ensure([], () => r(require("@/components/mainpage/user")), 'mainpage');
// const login = r => require.ensure([], () => r(require("@/login")), 'login');
// const register = r => require.ensure([], () => r(require("@/register")), 'register');
const setting = r => require.ensure([], () => r(require("@/components/mainpage/setting")), 'mainpage');
const contest = r => require.ensure([], () => r(require("@/components/mainpage/contest")), 'mainpage');
const contestdetail = r => require.ensure([], () => r(require("@/components/contest/contestdetail")), 'contest');
const problemdetail = r => require.ensure([], () => r(require("@/components/problem/problemdetail")), 'problem');
const rank = r => require.ensure([], () => r(require("@/components/mainpage/rank")), 'mainpage');
const admin = r => require.ensure([], () => r(require("@/components/mainpage/admin")), 'mainpage');
const billboard = r => require.ensure([], () => r(require("@/components/mainpage/billboard")), 'mainpage');
const blog = r => require.ensure([], () => r(require("@/components/mainpage/blog")), 'mainpage');
const wiki = r => require.ensure([], () => r(require("@/components/mainpage/wiki")), 'mainpage');
const algorithm = r => require.ensure([], () => r(require("@/components/wiki/algorithm")), 'wiki');
const mbcode = r => require.ensure([], () => r(require("@/components/wiki/code")), 'wiki');
const trainning = r => require.ensure([], () => r(require("@/components/wiki/trainning")), 'wiki');
const viewcode = r => require.ensure([], () => r(require("@/components/wiki/mbcode/viewcode")), 'wiki');
const viewcodedetail = r => require.ensure([], () => r(require("@/components/wiki/mbcode/viewcodedetail")), 'wiki');
const codeedit = r => require.ensure([], () => r(require("@/components/wiki/mbcode/codeedit")), 'wiki');
const wikidetail = r => require.ensure([], () => r(require("@/components/utils/wikidetail")), 'utils');
const trainningdetail = r => require.ensure([], () => r(require("@/components/wiki/trainning/trainningdetail")), 'wiki');
const newalgorithm = r => require.ensure([], () => r(require("@/components/wiki/newalgorithm")), 'wiki');
const todolist = r => require.ensure([], () => r(require("@/components/utils/todolist")), 'utils');
const homework = r => require.ensure([], () => r(require("@/components/mainpage/homework")), 'mainpage');
const givechoiceproblemscore = r => require.ensure([], () => r(require("@/components/admin/givechoiceproblemscore")), 'admin');
const classes = r => require.ensure([], () => r(require("@/components/mainpage/classes")), 'mainpage');
const classdetail = r => require.ensure([], () => r(require("@/components/mainpage/classdetail")), 'mainpage');


Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      name: 'homepage',
      component: homepage
    },
    // {
    //   path:'/login',
    //   name: 'login',
    //   component: login
    // },
    // {
    //   path:'/register',
    //   name: 'register',
    //   component: register
    // },
    {
      path: '/problem',
      name: 'problem',
      component: problem,
    },
    {
      path: '/problemdetail',
      name: 'problemdetail',
      component: problemdetail,
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
      path: '/wiki',
      name: 'wiki',
      component: wiki,
    },
    {
      path: '/classdetail',
      name: 'classdetail',
      component: classdetail,
    },
    {
      path: '/classes',
      name: 'classes',
      component: classes,
    },
    {
      path: '/wiki/algorithm',
      name: 'algorithm',
      component: algorithm,
    },
    {
      path: '/wiki/code',
      name: 'mbcode',
      component: mbcode,
    },
    {
      path: '/wiki/trainning',
      name: 'trainning',
      component: trainning,
    },
    {
      path: '/wiki/mbcode/:username',
      name: 'viewcode',
      component: viewcode,
    },
    {
      path: '/wiki/mbcodedetail/:codeID',
      name: 'viewcodedetail',
      component: viewcodedetail,
    },
    {
      path: '/wiki/mbcodeedit',
      name: 'codeedit',
      component: codeedit,
    },
    {
      path: '/wikidetail/:wikiid',
      name: 'wikidetail',
      component: wikidetail,
    },
    {
      path: '/trainningdetail/:trainningid',
      name: 'trainningdetail',
      component: trainningdetail,
    },
    {
      path: '/wiki/newalgorithm',
      name: 'newalgorithm',
      component: newalgorithm,
    },
    {
      path: '/todolist',
      name: 'todolist',
      component: todolist,
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
