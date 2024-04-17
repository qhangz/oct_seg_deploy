import request from '@/utils/request'

export function uploadImage(data: FormData) {
    return request({
        url: `/api/uploadimg`,
        method: 'POST',
        data,
        headers: {
            'Content-Type': 'multipart/form-data' // 设置请求头
        }
    })
}

export function getImage(image_id: string){
    return request({
        url: `/api/getimg/${image_id}`,
        method: 'GET',
        responseType: 'blob',
    })
}


export function clearImage() {
    return request({
        url: `/image`,
        method: 'DELETE',
    })
}
