# Load Balance

## Nginx

Nginx 是一个高性能的 HTTP 和反向代理服务器，具有负载均衡功能。它可以将客户端请求分发到多个后端服务器，以提高系统的可用性和性能。

## Location

在 Nginx 配置中，location 用于 匹配请求的 URI（路径）并定义如何处理这些请求。可以理解为一种请求路由规则：当客户端访问某个 URL 路径时，Nginx 根据 location 规则决定如何处理，例如转发到后端、返回文件或执行其他操作。

基本结构如下：

```bash
server {
    listen 80;

    location / {
        ...
    }
}
```
其中：

- server 定义一个虚拟主机（监听端口、域名等）。
- location 用于匹配请求路径。
- `{}` 内定义该路径的处理逻辑。

### Example

**Example 1**

```bash
location / {
    proxy_pass http://backend;
}
```

表示：所有路径都匹配都会进入这个规则，然后由 `proxy_pass` 转发到后端服务器。

**Example 2**

```bash
location /v1/ {
    proxy_pass http://backend;
}
```

只会匹配以 `/v1/` 开头的请求，例如 `/v1/completions`, `/v1/chat/completions` 等。

