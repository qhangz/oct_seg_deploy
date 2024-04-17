import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/home/index.vue'
import { message } from 'ant-design-vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('../views/login/index.vue'),
    },
    {
      path: '/seg',
      name: 'seg',
      component: HomeView,
      meta: {
        isAuth: true
      },
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue')
    },
    // login and register page
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/login/index.vue'),
      // meta: {
      //   title: '登录'
      // },
    },
    // page not found
    {
      path: '/404',
      name: '404',
      component: () => import('../views/404/index.vue'),
      meta: {
        title: 'page not found'
      },
    }
  ]
})

const loginMsg = () => {
  message.error('请先登录');

}
// 通过localStorage获取登录状态
router.beforeEach((to, from, next) => {
  if (to.meta.title) {
    document.title = to.meta.title as string ? to.meta.title : '加载中';
  }
  // next();
  if (to.meta.isAuth) { // 判断该路由是否需要登录权限
    if (localStorage.isLogin === 'true') {
      next();
    } else {
      next('/login')
      loginMsg()
    }
  } else {
    next();
  }
})

// if page not found
router.beforeEach((to, from, next) => {
  if (to.matched.length === 0) {
    // 路由不存在
    next('/404');
  } else {
    next();
  }
});


export default router
