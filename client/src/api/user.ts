import request from '@/utils/request'

// user register
export const userRegister = (username: string, password: any, email: any) => {
    return request({
        url: '/api/user/register',
        method: 'POST',
        data: {
            username: username,
            password: password,
            email: email,
        },
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    })
}

// user login
export const userLogin = (username: string, password: any) => {
    return request({
        url: '/api/user/login',
        method: 'POST',
        data: {
            username,
            password,
        },
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    })
}