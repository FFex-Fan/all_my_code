import {createRouter, createWebHistory} from 'vue-router'

import HomeView from '../views/HomeView.vue'
import LoginView from "@/views/LoginView.vue";
import NotFoundView from "@/views/NotFoundView.vue";
import RegisterView from "@/views/RegisterView.vue";
import UserActivityView from "@/views/UserActivityView.vue";
import UserListView from "@/views/UserListView.vue";

const routes = [
    {
        path: '/',
        name: 'home',
        component: HomeView
    },
    {
        path: '/login/',
        name: 'login',
        component: LoginView
    },
    {
        path: '/404/',
        name: '404',
        component: NotFoundView
    },
    {
        path: '/register/',
        name: 'register',
        component: RegisterView
    },
    {
        // 使不同用户可以对应到不同的界面
        path: '/user_activity/:user_id/',
        name: 'user_activity',
        component: UserActivityView
    },
    {
        path: '/user_list/',
        name: 'user_list',
        component: UserListView
    },
    {
        // 正则表达式匹配 :catchAll(),   . 表示匹配任意字符，* 表示匹配任意长度(即匹配任意字符)
        path: '/:catchAll(.*)',
        redirect: '/404/',  // 重定向到 404 页面
    },
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router
