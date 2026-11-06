# Agent 的指令与工具

同一句“你好！”，发给写作助手和代码助手，模型收到的输入可以很不一样。区别来自应用在发送请求前做的两件事：选取本轮需要遵循的指令，声明模型可以调用的工具。

## 指令从哪里来

写作助手的指令可以直接放在应用配置里。例如，配置里写着“修改草稿时保留作者的术语”。应用构造请求时，把这段文字作为系统消息，把你输入的“你好！”作为用户消息。两段文字都进入模型的上下文，角色标记让模型能区分应用的要求和用户的话。

项目级指令需要先确定作用范围。以 OpenAI 公布的 Codex CLI 实现为例，它会读取用户级指令，并从项目根目录沿路径走到当前工作目录，收集适用的 `AGENTS.md` 等指令文件。越接近当前目录的指令越靠后。如果你在 `docs/zh/` 工作，仓库根目录可以约定文档的通用格式，`docs/zh/AGENTS.md` 则可以补充中文术语的写法。CLI 将收集到的指令和工作环境信息放入首轮输入，再追加你键入的消息。在这个实现里，汇总的项目指令随一条 `user` 角色消息传入，因此同为 `user` 角色的内容也可能由应用生成。[Codex CLI 首轮请求示例](https://openai.com/index/unrolling-the-codex-agent-loop/)

## 工具如何进入请求

假设写作助手可以按标题读取已保存的草稿。在 Chat Completions 风格的 API 中，应用通过请求顶层的 `tools` 字段声明这个能力。下面是工具定义的一个例子。

```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "read_draft",
      "description": "按标题读取用户保存的草稿，返回正文。",
      "parameters": {
        "type": "object",
        "properties": {
          "title": {"type": "string", "description": "草稿标题"}
        },
        "required": ["title"]
      }
    }
  }]
}
```

模型从名称和描述判断何时该用这个工具，从参数定义得知调用时需要提供草稿标题。对 GLM-5.2，服务端的对话模板会把工具定义编入 prompt，再交给 tokenizer。因此，只要本轮声明了这个工具，它的定义就可能占用输入 token，即使用户只说了“你好！”。[GLM-5.2 的对话模板](https://huggingface.co/zai-org/GLM-5.2/blob/main/chat_template.jinja)

工具定义和工具结果在不同轮次进入请求。假设你接着说“帮我润色《序言》的第二段”，而这篇草稿还没有出现在对话里。模型可以先生成对 `read_draft` 的调用，给出参数 `{"title":"序言"}`。写作助手执行读取，把返回的正文和对应的调用记录加入下一次模型请求。到这一轮，模型才有第二段的原文可以修改。[Agent 工具调用循环](https://openai.com/index/unrolling-the-codex-agent-loop/)

## 工具定义有多长

Anthropic 给过一个具体配置：连接 GitHub、Slack、Sentry、Grafana 和 Splunk 五个 MCP 服务，共 58 个工具。一次性加载全部定义时，仅这部分就约占 5.5 万个输入 token。对于“你好！”这样的问候，消息本身很短，决定输入长度的反而可能是应用预先声明了哪些工具。[Anthropic 的工具加载示例](https://www.anthropic.com/research/advanced-tool-use)

同一篇文章介绍的 Tool Search 把加载推迟到需要工具时。模型先看到一个搜索工具。等你要求读取草稿，它可以搜索并取得相关定义，再发起调用。两种做法的取舍很具体：预先加载会扩大首轮输入，按需加载则可能多一次搜索。要确定某个 Agent 的实际开销，需要看它在那一轮发送了哪些指令、消息和工具定义，以及该次请求记录的输入 token 数。
