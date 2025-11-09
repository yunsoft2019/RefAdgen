import datetime
from PIL import Image


class BaseUtils:
    @staticmethod
    def seconds_to_hms(seconds):
        hours = int(seconds // 3600)
        remaining_seconds = seconds % 3600
        minutes = int(remaining_seconds // 60)
        secs = int(remaining_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def calculate_future_time(seconds):
        now = datetime.datetime.now()
        future_time = now + datetime.timedelta(seconds=seconds)
        if future_time.date() == now.date():
            day_str = "今天"
        elif future_time.date() == now.date() + datetime.timedelta(days=1):
            day_str = "明天"
        elif future_time.date() == now.date() + datetime.timedelta(days=2):
            day_str = "后天"
        else:
            return "超出计算范围"

        time_str = future_time.strftime("%H:%M")
        result = f"{day_str}:{time_str}"
        return result

    @staticmethod
    def resize_image_get_height(image_path, new_width):
        image = Image.open(image_path)
        width, height = image.size
        ratio = new_width / width
        new_height = int(height * ratio)
        return new_height

