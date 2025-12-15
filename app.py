import gradio as gr
from service import Service
from logger_config import logger_app

def doctor_bot(message, history):
    logger_app.debug(f"doctor_bot() - 收到消息: {message}")
    logger_app.debug(f"doctor_bot() - 历史记录: {len(history)} 条")

    try:
        logger_app.debug("doctor_bot() - 创建Service实例")
        service = Service()

        logger_app.info("doctor_bot() - 开始处理问题")
        result = service.answer(message, history)

        logger_app.info("doctor_bot() - 问题处理完成")
        logger_app.debug(f"doctor_bot() - 回答长度: {len(result)} 字符")
        return result
    except Exception as e:
        logger_app.error(f"doctor_bot() - 处理失败: {e}")
        return "抱歉，系统暂时无法回答您的问题，请稍后重试。"

css = '''
.gradio-container { max-width:850px !important; margin:20px auto !important;}
.message { padding: 10px !important; font-size: 14px !important;}
'''

demo = gr.ChatInterface(
    css = css,
    fn = doctor_bot,
    title = '医疗问诊机器人',
    chatbot = gr.Chatbot(height=400, bubble_full_width=False),
    theme = gr.themes.Default(spacing_size='sm', radius_size='sm'),
    textbox=gr.Textbox(placeholder="在此输入您的问题", container=False, scale=7),
    examples = ['你好，你叫什么名字？', '寻医问药网获得过哪些投资？', '寻医问药网获的客服电话是多少？', '鼻炎是一种什么病？', '一般会有哪些症状？', '吃什么药好得快？可以吃阿莫西林吗？', '刀郎最近有什么新专辑？'],
    submit_btn = gr.Button('提交', variant='primary'),
    # clear_btn = gr.Button('清空记录'),
    # retry_btn = None,
    # undo_btn = None,
)

if __name__ == '__main__':
    logger_app.info("app.py - 启动医疗问诊机器人应用")
    logger_app.debug("app.py - Gradio配置: max_width=850px")

    try:
        demo.launch(share=True)
        logger_app.info("app.py - 应用启动成功")
    except Exception as e:
        logger_app.error(f"app.py - 应用启动失败: {e}")