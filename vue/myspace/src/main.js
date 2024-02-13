import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap/dist/js/bootstrap'
// import 'element-plus/dist/index.css'
// import 'element-plus/dist/index.full.js'


createApp(App).use(store).use(router).mount('#app')
