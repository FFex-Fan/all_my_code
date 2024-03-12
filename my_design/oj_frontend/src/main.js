import { createApp } from 'vue'
import App from './App.vue'
import store from './store'
import router from './router'

// import useClipboard from 'vue-clipboard3'

import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap/dist/js/bootstrap'
import 'bootstrap-icons/font/bootstrap-icons.css'

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'



createApp(App).use(router).use(router).use(store).use(ElementPlus).mount('#app')
