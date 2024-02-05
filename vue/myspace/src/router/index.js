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
        path: '/login',
        name: 'login',
        component: LoginView
    },
    {
        path: '/404',
        name: '404',
        component: NotFoundView
    },
    {
        path: '/register',
        name: 'register',
        component: RegisterView
    },
    {
        path: '/user_activity',
        name: 'user_activity',
        component: UserActivityView
    },
    {
        path: '/user_list',
        name: 'user_list',
        component: UserListView
    },
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router
