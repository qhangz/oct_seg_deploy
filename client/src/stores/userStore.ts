// 管理用户数据相关
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { userLogin } from '@/api/user'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue';

const loginSuccessMsg = () => {
    message.success('登录成功');
}
const loginWarningMsg = () => {
    message.warning('登录有误');
}
const logoutSuccessMsg = () => {
    message.success('退出登录');
}

const alreadyLoginMsg = () => {
    message.warning('用户已登录');
}
const userNotFoundMsg = () => {
    message.error('用户不存在');
}
const errorPsw = () => {
    message.error('密码错误');
}
const errorMsg = () => {
    message.error('出错了');
}
export const useUserStore = defineStore('user', () => {
    const router = useRouter()
    // 管理用户登录状态的state
    const userState = {
        isLogin: false,  //是否登录
        // userInfo: {},    //用户信息
        // token: '',       //用户token
    }
    // 管理用户数据的state
    let userInfo = ref({})

    // 登录
    const login = async ({ username, password }: { username: string, password: any }) => {
        try {
            if (userState.isLogin === true) {
                // already login
                alreadyLoginMsg()   //提示已经登录
            } else {
                const res = await userLogin(username, password)
                if (res.code == 200) {
                    loginSuccessMsg()   //提示登录成功
                    // userState.token = res.data.token
                    userState.isLogin = true
                    // userInfo.value = res.data.userInfo
                    // console.log("1", userInfo.value);
                    // localstorage存储登录状态
                    localStorage.setItem('isLogin', 'true')
                    // localStorage.setItem('token', res.data.token)
                    // localStorage.setItem('userInfo', JSON.stringify(userInfo.value))
                    // 登录成功后，跳转到seg面
                    router.push('/seg')
                } else if (res.code == 400) {
                    if (res.msg == 'user not found') {
                        userNotFoundMsg()   //提示用户不存在
                    } else if (res.msg == 'password error') {
                        errorPsw()   //提示密码有误
                    }else{
                        errorMsg()  //提示登录有误
                    }
                } else {
                    errorMsg()  //提示登录有误
                }
            }
        } catch (error) {
            console.log(error);
            loginWarningMsg()
        }
    }


    // 登出
    const logout = () => {
        logoutSuccessMsg()  //提示退出登录
        userState.isLogin = false
        // userState.token = ''
        userInfo.value = {}
        localStorage.removeItem('isLogin')
        // localStorage.removeItem('token')
        // localStorage.removeItem('userInfo')
    }

    // 刷新页面时，从localstorage中获取登录状态
    const isUserLogin = () => {
        const isLogin = localStorage.getItem('isLogin')
        // const token = localStorage.getItem('token')
        // const userInfo = localStorage.getItem('userInfo')
        if (isLogin) {
            userState.isLogin = true
            // userState.token = token || ''
            // userInfo.value = userInfo ? JSON.parse(userInfo) : {}
            // userInfo.value = userInfo ? JSON.parse(userInfo) : {}
        } else {
            userState.isLogin = false
        }
    }

    // 改变登录状态
    const changeLoginState = (state: boolean) => {
        userState.isLogin = state
    }

    // 获取接口数据的action函数
    const getUserInfo = async ({ username, password }: { username: string, password: any }) => {
        const res = await userLogin(username, password)
        userInfo.value = res
    }
    // 退出时清除用户信息
    const clearUserInfo = () => {
        userState.isLogin = false
        // userState.token = ''
        userInfo.value = {}
        localStorage.removeItem('isLogin')
        // localStorage.removeItem('token')
        // localStorage.removeItem('userInfo')
    }

    return {
        userState,
        userInfo,
        login,
        logout,
        isUserLogin,
        changeLoginState,
        getUserInfo,
        clearUserInfo,
    }
})
