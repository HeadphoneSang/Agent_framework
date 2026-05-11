import requests


def get_bilibili_video(bvid, cookies=None):
    """
    根据B站视频BV号下载视频
    :param bvid: B站视频BV号
    :param cookies: 必须包含SESSDATA和bili_jct的字典
    """
    # 强制校验bili_jct字段
    if cookies is None or 'bili_jct' not in cookies:
        raise ValueError("cookies参数必须包含bili_jct字段")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"https://www.bilibili.com/video/{bvid}",
        "Origin": "https://www.bilibili.com",
        "Connection": "keep-alive"
    }

    # 获取视频基础信息
    info_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    info_resp = requests.get(info_url, headers=headers, cookies=cookies)
    if info_resp.status_code != 200:
        raise Exception(f"视频信息获取失败: HTTP {info_resp.status_code}")

    info_data = info_resp.json()
    if info_data['code'] != 0:
        raise Exception(f"API错误: {info_data['message']}")

    # 校验pages列表非空
    pages = info_data["data"].get("pages", [])
    if not pages:
        raise Exception("视频分页信息为空，无法获取cid")

    cid = pages[0]["cid"]

    # 动态获取有效清晰度
    accept_quality = info_data["data"].get("accept_quality", [])
    if not accept_quality:
        print("未找到有效清晰度，尝试获取有效清晰度")
        accept_quality = [480]

    # 尝试获取有效视频流
    video_data = None
    for qn in accept_quality:
        params = {
            "bvid": bvid,
            "cid": cid,
            "qn": qn,
            "platform": "html5",
            "csrf": cookies['bili_jct']
        }
        play_resp = requests.get(
            "https://api.bilibili.com/x/player/playurl",
            params=params,
            headers=headers,
            cookies=cookies
        )
        if play_resp.status_code == 200 and play_resp.json().get('code') == 0:
            video_data = play_resp.json()['data']
            break

    if not video_data:
        raise Exception("未找到有效视频流")

    # 处理DASH格式和多分片
    if 'dash' in video_data:
        # 优先处理DASH格式
        dash = video_data['dash']
        video_streams = dash.get('video', [])
        audio_streams = dash.get('audio', [])

        # 选择最高带宽的视频流
        if not video_streams:
            raise Exception("DASH中未找到视频流")
        video_stream = max(video_streams, key=lambda x: x['bandwidth'])
        video_url_list = [video_stream['baseUrl']]

        # 选择最高带宽的音频流（如有）
        audio_url_list = []
        if audio_streams:
            audio_stream = max(audio_streams, key=lambda x: x['bandwidth'])
            audio_url_list = [audio_stream['baseUrl']]

        # 合并音视频处理逻辑（需ffmpeg）
        print(f"检测到DASH格式，包含{len(video_url_list)}个视频流和{len(audio_url_list)}个音频流")
    else:
        # 处理传统durl格式
        video_url_list = [d['url'] for d in video_data['durl']]
        audio_url_list = []
        print(f"检测到{len(video_url_list)}个视频分片")

    # 合并下载分片
    output_file = f"{bvid}.mp4"
    with open(output_file, "wb") as f:
        # 下载视频流
        for idx, url in enumerate(video_url_list, 1):
            print(f"正在下载视频分片 {idx}/{len(video_url_list)}")
            chunk_resp = requests.get(url, headers=headers, stream=True)
            for chunk in chunk_resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # 下载音频流（如有）
        for idx, url in enumerate(audio_url_list, 1):
            print(f"正在下载音频分片 {idx}/{len(audio_url_list)}")
            chunk_resp = requests.get(url, headers=headers, stream=True)
            for chunk in chunk_resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"视频已保存至: {output_file}")
    return output_file

if __name__ == '__main__':
    file_0 = get_bilibili_video('BV1MUmEBJELw',cookies={
        'SESSDATA': '537b31d8%2C1793867272%2C8cacc%2A52CjDFeUkVg6p-HZlFntJntt3AtClQ2gtwh0wLxN4CFEnzhdmZWHVJcdpaQRQM6DzT4TISVm5NR0JfcGRleS1VN1NxMkhseUZoUklEcnVucGpzZ0FuZV85eEtCb3MyejZwNVlqSHNVYllKdkZaY3gxemRoTnM4dE94RlEyc0ZwS0RRUzdOZFBDNG5RIIEC',
        'bili_jct': '4f5deaa309cbdbacb0fa855e2cedf275'}

    )
    print(file_0)