
/**
 * GET
 */

var get = (url, args = {}) => new Promise((resolve, reject) => {
    var xhr = new XMLHttpRequest()
    xhr.open('GET', url, true)
    for (var key in args.headers)
        xhr.setRequestHeader(key, args.headers[key])
    xhr.onreadystatechange = () => {
        if (xhr.readyState != 4)
            return
        if (xhr.status == 200) {
            let res = {}
            res.xhr = xhr
            res.headers = {}
            for (let line in xhr.getAllResponseHeaders().split('\n')) {
                if (!line.trim().length) continue
                let [key, value] = line.split(': ')
                res.headers[key] = (value === undefined ? '' : value)
            }
            res.text = xhr.responseText
            res.body = xhr.response
            res.status = xhr.status
            resolve(res)
        } else { reject(xhr) }
    }
    xhr.send()
})

/**
 * POST
 */

var post = (url, args = {}) => new Promise((resolve, reject) => {
    var xhr = new XMLHttpRequest()
    var body = args.data
    xhr.open('POST', url, true)
    if (args.form !== undefined) {
        xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded')
        if (typeof args.form == 'string')
            body = args.form
        else if (typeof args.form == 'object')
            body = Object.keys(args.form)
                .map(key => [key, args.form[key]].map(encodeURIComponent).join('='))
                .join('&')
    }
    else if (args.json !== undefined) {
        xhr.setRequestHeader('Content-type', 'application/json')
        body = JSON.stringify(args.json)
    }
    for (var key in args.headers)
        xhr.setRequestHeader(key, args.headers[key])
    xhr.onreadystatechange = () => {
        if (xhr.readyState != 4)
            return
        if (xhr.status == 200) {
            let res = {}
            res.xhr = xhr
            res.headers = {}
            for (let line in xhr.getAllResponseHeaders().split('\n')) {
                if (!line.trim().length) continue
                let [key, value] = line.split(': ')
                res.headers[key] = (value === undefined ? '' : value)
            }
            res.text = xhr.responseText
            res.body = xhr.response
            res.status = xhr.status
            resolve(res)
        } else { reject(xhr) }
    }
    xhr.send(body)
})

/**
 * GET Minified
 */

var get=(a,b={})=>new Promise((c,d)=>{var e=new XMLHttpRequest;for(var f in e.open('GET',a,!0),b.headers)e.setRequestHeader(f,b.headers[f]);e.onreadystatechange=()=>{if(4==e.readyState)if(200==e.status){let g={xhr:e,headers:{}};for(let h in e.getAllResponseHeaders().split('\n'))if(h.trim().length){let[k,l]=h.split(': ');g.headers[k]=void 0===l?'':l}g.text=e.responseText,g.body=e.response,g.status=e.status,c(g)}else d(e)},e.send()});

/**
 * POST Minified
 */

var post=(a,b={})=>new Promise((c,d)=>{var e=new XMLHttpRequest,f=b.data;for(var g in e.open('POST',a,!0),void 0===b.form?void 0!==b.json&&(e.setRequestHeader('Content-type','application/json'),f=JSON.stringify(b.json)):(e.setRequestHeader('Content-type','application/x-www-form-urlencoded'),'string'==typeof b.form?f=b.form:'object'==typeof b.form&&(f=Object.keys(b.form).map(h=>[h,b.form[h]].map(encodeURIComponent).join('=')).join('&'))),b.headers)e.setRequestHeader(g,b.headers[g]);e.onreadystatechange=()=>{if(4==e.readyState)if(200==e.status){let h={xhr:e,headers:{}};for(let i in e.getAllResponseHeaders().split('\n'))if(i.trim().length){let[l,m]=i.split(': ');h.headers[l]=void 0===m?'':m}h.text=e.responseText,h.body=e.response,h.status=e.status,c(h)}else d(e)},e.send(f)});
