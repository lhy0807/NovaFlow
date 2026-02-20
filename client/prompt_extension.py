# adapted from https://github.com/Wan-Video/Wan2.1

import argparse
from google import genai
from google.genai import types

def generate(image1_path: str, prompt: str, image2_path: str | None = None) -> str:
    client = genai.Client()

    parts = []
    parts.append(
        types.Part.from_bytes(
            data=open(image1_path, "rb").read(),
            mime_type="image/png",
        )
    )

    if image2_path:
        parts.append(
            types.Part.from_bytes(
                data=open(image2_path, "rb").read(),
                mime_type="image/png",
            )
        )

    parts.append(types.Part.from_text(text=prompt))

    multi_image_prompt = """你是一位Prompt优化师，旨在参考用户输入的图像的细节内容，把用户输入的Prompt改写为优质Prompt，使其更完整、更具表现力，同时不改变原意。你需要综合用户输入的照片内容和输入的Prompt进行改写，严格参考示例的格式进行改写
任务要求：
1. 用户会输入两张图片，第一张是视频的第一帧，第二张时视频的最后一帧，你需要综合两个照片的内容进行优化改写
2. 对于过于简短的用户输入，在不改变原意前提下，合理推断并补充细节，使得画面更加完整好看；
3. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等）、画面风格、空间关系、镜头景别；
4. 整体中文输出，保留引号、书名号中原文以及重要的输入信息，不要改写；
5. Prompt应匹配符合用户意图且精准细分的风格描述。如果用户未指定，则根据用户提供的照片的风格，你需要仔细分析照片的风格，并参考风格进行改写。
6. 如果Prompt是古诗词，应该在生成的Prompt中强调中国古典元素，避免出现西方、现代、外国场景；
7. 你需要强调输入中的运动信息和不同的镜头运镜；
8. 你的输出应当带有自然运动属性，需要根据描述主体目标类别增加这个目标的自然动作，描述尽可能用简单直接的动词；
9. 你需要尽可能的参考图片的细节信息，如人物动作、服装、背景等，强调照片的细节元素；
10. 你需要强调两画面可能出现的潜在变化，如“走进”，“出现”，“变身成”，“镜头左移”，“镜头右移动”，“镜头上移动”， “镜头下移”等等；
11. 无论用户输入那种语言，你都需要输出中文；
12. 改写后的prompt字数控制在80-100字左右；
改写后 prompt 示例：
1. 日系小清新胶片写真，扎着双麻花辫的年轻东亚女孩坐在船边。女孩穿着白色方领泡泡袖连衣裙，裙子上有褶皱和纽扣装饰。她皮肤白皙，五官清秀，眼神略带忧郁，直视镜头。女孩的头发自然垂落，刘海遮住部分额头。她双手扶船，姿态自然放松。背景是模糊的户外场景，隐约可见蓝天、山峦和一些干枯植物。复古胶片质感照片。中景半身坐姿人像。
2. 二次元厚涂动漫插画，一个猫耳兽耳白人少女手持文件夹，神情略带不满。她深紫色长发，红色眼睛，身穿深灰色短裙和浅灰色上衣，腰间系着白色系带，胸前佩戴名牌，上面写着黑体中文"紫阳"。淡黄色调室内背景，隐约可见一些家具轮廓。少女头顶有一个粉色光圈。线条流畅的日系赛璐璐风格。近景半身略俯视视角。
3. CG游戏概念数字艺术，一只巨大的鳄鱼张开大嘴，背上长着树木和荆棘。鳄鱼皮肤粗糙，呈灰白色，像是石头或木头的质感。它背上生长着茂盛的树木、灌木和一些荆棘状的突起。鳄鱼嘴巴大张，露出粉红色的舌头和锋利的牙齿。画面背景是黄昏的天空，远处有一些树木。场景整体暗黑阴冷。近景，仰视视角。
4. 美剧宣传海报风格，身穿黄色防护服的Walter White坐在金属折叠椅上，上方无衬线英文写着"Breaking Bad"，周围是成堆的美元和蓝色塑料储物箱。他戴着眼镜目光直视前方，身穿黄色连体防护服，双手放在膝盖上，神态稳重自信。背景是一个废弃的阴暗厂房，窗户透着光线。带有明显颗粒质感纹理。中景，镜头下移。
请直接输出改写后的文本，不要进行多余的回复。"""
    single_image_prompt = '''你是一位Prompt优化师，旨在参考用户输入的图像的细节内容，把用户输入的Prompt改写为优质Prompt，使其更完整、更具表现力，同时不改变原意。你需要综合用户输入的照片内容和输入的Prompt进行改写，严格参考示例的格式进行改写。\n''' \
    '''任务要求：\n''' \
    '''1. 对于过于简短的用户输入，在不改变原意前提下，合理推断并补充细节，使得画面更加完整好看；\n''' \
    '''2. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等）、画面风格、空间关系、镜头景别；\n''' \
    '''3. 整体中文输出，保留引号、书名号中原文以及重要的输入信息，不要改写；\n''' \
    '''4. Prompt应匹配符合用户意图且精准细分的风格描述。如果用户未指定，则根据用户提供的照片的风格，你需要仔细分析照片的风格，并参考风格进行改写；\n''' \
    '''5. 如果Prompt是古诗词，应该在生成的Prompt中强调中国古典元素，避免出现西方、现代、外国场景；\n''' \
    '''6. 你需要强调输入中的运动信息和不同的镜头运镜；\n''' \
    '''7. 你的输出应当带有自然运动属性，需要根据描述主体目标类别增加这个目标的自然动作，描述尽可能用简单直接的动词；\n''' \
    '''8. 你需要尽可能的参考图片的细节信息，如人物动作、服装、背景等，强调照片的细节元素；\n''' \
    '''9. 改写后的prompt字数控制在80-100字左右\n''' \
    '''10. 无论用户输入什么语言，你都必须输出中文\n''' \
    '''改写后 prompt 示例：\n''' \
    '''1. 日系小清新胶片写真，扎着双麻花辫的年轻东亚女孩坐在船边。女孩穿着白色方领泡泡袖连衣裙，裙子上有褶皱和纽扣装饰。她皮肤白皙，五官清秀，眼神略带忧郁，直视镜头。女孩的头发自然垂落，刘海遮住部分额头。她双手扶船，姿态自然放松。背景是模糊的户外场景，隐约可见蓝天、山峦和一些干枯植物。复古胶片质感照片。中景半身坐姿人像。\n''' \
    '''2. 二次元厚涂动漫插画，一个猫耳兽耳白人少女手持文件夹，神情略带不满。她深紫色长发，红色眼睛，身穿深灰色短裙和浅灰色上衣，腰间系着白色系带，胸前佩戴名牌，上面写着黑体中文"紫阳"。淡黄色调室内背景，隐约可见一些家具轮廓。少女头顶有一个粉色光圈。线条流畅的日系赛璐璐风格。近景半身略俯视视角。\n''' \
    '''3. CG游戏概念数字艺术，一只巨大的鳄鱼张开大嘴，背上长着树木和荆棘。鳄鱼皮肤粗糙，呈灰白色，像是石头或木头的质感。它背上生长着茂盛的树木、灌木和一些荆棘状的突起。鳄鱼嘴巴大张，露出粉红色的舌头和锋利的牙齿。画面背景是黄昏的天空，远处有一些树木。场景整体暗黑阴冷。近景，仰视视角。\n''' \
    '''4. 美剧宣传海报风格，身穿黄色防护服的Walter White坐在金属折叠椅上，上方无衬线英文写着"Breaking Bad"，周围是成堆的美元和蓝色塑料储物箱。他戴着眼镜目光直视前方，身穿黄色连体防护服，双手放在膝盖上，神态稳重自信。背景是一个废弃的阴暗厂房，窗户透着光线。带有明显颗粒质感纹理。中景人物平视特写。\n''' \
    '''直接输出改写后的文本。'''

    if image2_path:
        si_text1 = multi_image_prompt
    else:
        si_text1 = single_image_prompt

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=parts,
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        max_output_tokens=65535,
        system_instruction=[types.Part.from_text(text=si_text1)],
        thinking_config=types.ThinkingConfig(
            thinking_budget=1024,
        ),
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text or prompt


def main():
    parser = argparse.ArgumentParser(description="Generate text from images and a prompt.")
    parser.add_argument("--image1", required=True, help="Path to the first image.")
    parser.add_argument(
        "--image2", help="Path to the second image."
    )
    parser.add_argument("--prompt", required=True, help="The text prompt.")
    args = parser.parse_args()

    print(generate(args.image1, args.prompt, args.image2))


if __name__ == "__main__":
    main()