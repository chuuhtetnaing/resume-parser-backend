import base64
import re

import cv2
import spacy
from transformers import pipeline

skill_ner = spacy.load("en_core_web_lg")
skill_pattern_path = "data/jz_skill_patterns.jsonl"
ruler = skill_ner.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)

address_pipe = pipeline(
    "token-classification",
    model="DioulaD/birdi-finetuned-ner-address-v2",
    grouped_entities=True,
)
education_pipe = pipeline(
    "token-classification",
    model="Jean-Baptiste/camembert-ner-with-dates",
    grouped_entities=True,
)


def filter_ocr_data(section_bbox, ocr_data):
    section_bbox = {
        "x1": int(section_bbox[0]),
        "y1": int(section_bbox[1]),
        "x2": int(section_bbox[2]),
        "y2": int(section_bbox[3]),
    }

    def get_iou(other_bbox):
        x_left = max(section_bbox["x1"], other_bbox["x1"])
        y_top = max(section_bbox["y1"], other_bbox["y1"])
        x_right = min(section_bbox["x2"], other_bbox["x2"])
        y_bottom = min(section_bbox["y2"], other_bbox["y2"])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        other_bbox_area = (other_bbox["x2"] - other_bbox["x1"]) * (
            other_bbox["y2"] - other_bbox["y1"]
        )

        if other_bbox_area == 0:
            return 0.0

        iou = intersection_area / other_bbox_area

        return iou

    ocr_result = list()
    for row in ocr_data:
        bbox = row["bbox"]
        text = row["text"]
        block_no = row["block_no"]

        word_bbox = {
            "x1": int(bbox[0]),
            "y1": int(bbox[1]),
            "x2": int(bbox[2]),
            "y2": int(bbox[3]),
        }

        iou_status = get_iou(word_bbox)
        if iou_status >= 0.2:
            ocr_result.append({"text": text, "bbox": bbox, "block_no": block_no})

    return ocr_result


def get_section(layout, section_name):
    result = list(filter(lambda x: x["label"].lower() == section_name.lower(), layout))
    return result


class ResumeParser:
    def __init__(self, layout_result, first_page_cv2):
        self.layout_result = layout_result
        self.first_page_cv2 = first_page_cv2

    def get_name(self):
        name_sections = get_section(self.layout_result["layout"], "name")
        if len(name_sections) == 0:
            return False

        name_section = name_sections[0]
        names = filter_ocr_data(name_section["bbox"], self.layout_result["line_ocr"])

        if len(names) == 0:
            return False

        return {"full_name": names[0]["text"]}

    def get_profile_image(self):
        images = get_section(self.layout_result["layout"], "image")
        if len(images) == 0:
            return False

        image_bbox = images[0]["bbox"]
        cropped_image = self.first_page_cv2[
            image_bbox[1] : image_bbox[3], image_bbox[0] : image_bbox[2]
        ]
        img_bytes = cv2.imencode(".png", cropped_image)[1].tobytes()
        base64_image_data = base64.b64encode(img_bytes).decode("utf-8")
        return {"picture_base64": base64_image_data}

    def get_objective(self):
        objective_sections = get_section(self.layout_result["layout"], "resume")
        if len(objective_sections) == 0:
            return False

        objective_ocr = list()

        for section in objective_sections:
            objective_ocr.extend(
                filter_ocr_data(section["bbox"], self.layout_result["line_ocr"])
            )

        if len(objective_ocr) == 0:
            return False

        objective_text = " ".join([objective["text"] for objective in objective_ocr])
        raw_objective_text = re.sub("[\s\n]+", " ", objective_text)
        return {"description": raw_objective_text}

    def get_contact(self):
        phone_regex = re.compile(
            r"""(
        [\+]?[(]?[\+]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,7}
         )""",
            re.VERBOSE,
        )

        email_regex = re.compile(
            r"""
        [a-zA-Z0-9_.+-]+ # name part
        @ # @ symbol
        [a-zA-Z0-9_.+-]+ # domain part
        """,
            re.VERBOSE,
        )

        contact_sections = get_section(self.layout_result["layout"], "contact")
        if len(contact_sections) == 0:
            return False

        contact_ocr = list()

        for section in contact_sections:
            contact_ocr.extend(
                filter_ocr_data(section["bbox"], self.layout_result["line_ocr"])
            )

        if len(contact_ocr) == 0:
            return False

        contact_ocr_line = " ".join([line["text"] for line in contact_ocr])
        raw_contact_text = re.sub("[\s\n]+", " ", contact_ocr_line)

        extracted_phones = phone_regex.findall(raw_contact_text)
        extracted_phone = extracted_phones[0] if len(extracted_phones) else ""

        extracted_emails = email_regex.findall(raw_contact_text)
        extracted_email = extracted_emails[0] if len(extracted_emails) else ""

        address_entities = filter(
            lambda entity: entity["entity_group"] == "ADDRESS",
            address_pipe(raw_contact_text),
        )
        address_words = map(lambda x: x["word"], address_entities)
        address = " ".join(address_words)

        return {
            "email": [{"value": extracted_email}],
            "phone": [{"type": "Telephone", "value": extracted_phone}],
            "address": [{"value": address}],
        }

    def get_skills(self):
        all_lines = " ".join([line["text"] for line in self.layout_result["line_ocr"]])
        raw_all_lines = re.sub("[\s\n]+", " ", all_lines)

        doc = skill_ner(raw_all_lines)
        skills = []
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                skills.append(ent.text.lower())
        result = list(set(skills))
        return result

    def get_education(self):
        education_sections = get_section(self.layout_result["layout"], "education")
        if len(education_sections) == 0:
            return False

        education_ocr = list()

        for section in education_sections:
            education_ocr.extend(
                filter_ocr_data(section["bbox"], self.layout_result["line_ocr"])
            )

        if len(education_ocr) == 0:
            return False

        education_list = [education["text"] for education in education_ocr]
        education_text = " ".join(education_list)
        raw_education_text = re.sub("[\s\n\(\)]+", " ", education_text)

        dates = list()
        universities = list()
        degrees = list()

        entities = education_pipe(raw_education_text)

        for entity in entities:
            if len(entity["word"].split()) == 1:
                continue

            if entity["entity_group"] == "DATE":
                dates.append(entity["word"])

            if entity["entity_group"] == "MISC":
                degrees.append(entity["word"])

            if entity["entity_group"] == "ORG":
                universities.append(entity["word"])

        educations = list()
        for university, date_, degree in zip(universities, dates, degrees):
            education = {"school": university, "date": date_, "degree_name": degree}
            educations.append(education)

        return educations

    def get_experience(self):
        experience_sections = get_section(self.layout_result["layout"], "experience")

        if len(experience_sections) == 0:
            return False

        experience_ocr = list()

        for section in experience_sections:
            experience_ocr.extend(
                filter_ocr_data(section["bbox"], self.layout_result["line_ocr"])
            )

        if len(experience_ocr) == 0:
            return False

        def group_ocr_by_block(experience_ocr):
            result = {}
            for item in experience_ocr:
                block_no = item["block_no"]
                text = item["text"]
                bbox = item["bbox"]

                if block_no in result:
                    result[block_no]["text"] += " " + text
                    bbox1 = result[block_no]["bbox"]
                    result[block_no]["bbox"] = [
                        min(bbox1[0], bbox[0]),
                        min(bbox1[1], bbox[1]),
                        max(bbox1[2], bbox[2]),
                        max(bbox1[3], bbox[3]),
                    ]
                else:
                    result[block_no] = {"text": text, "bbox": bbox}

            return list(result.values())

        grouped_experience_ocr = group_ocr_by_block(experience_ocr)

        min_start_x = min([ocr["bbox"][0] for ocr in grouped_experience_ocr])

        experiences = list()
        current_experience = dict()

        title_indexes = list()
        for i, ocr in enumerate(grouped_experience_ocr):
            if min_start_x - 20 < ocr["bbox"][0] < min_start_x + 20:
                if len(current_experience) != 0:
                    experiences.append(current_experience.copy())

                title_indexes.append(i)

        def group_continuous_items(items):
            grouped_items = []
            current_group = []

            for item in items:
                if not current_group or item == current_group[-1] + 1:
                    current_group.append(item)
                else:
                    grouped_items.append(current_group)
                    current_group = [item]

            if current_group:
                grouped_items.append(current_group)

            return grouped_items

        titles = group_continuous_items(title_indexes)

        description_ranges = list()
        total = len(titles)
        for index, item in enumerate(titles):
            current_biggest_index = max(item)
            next_start_index = min(
                current_biggest_index + 1, len(grouped_experience_ocr) - 1
            )

            if index == total - 1:
                description_ranges.append(
                    range(next_start_index, len(grouped_experience_ocr) + 1)
                )
            else:
                next_smallest_index = min(titles[index + 1])
                next_end_index = next_smallest_index + 1

                description_ranges.append(range(next_start_index, next_end_index))

        separated_title_description = list()

        for title_range, description_range in zip(titles, description_ranges):
            title_list = grouped_experience_ocr[title_range[0] : title_range[-1] + 1]
            description_list = grouped_experience_ocr[
                list(description_range)[0] : list(description_range)[-1]
            ]

            raw_titles = " ".join([title["text"] for title in title_list])
            raw_descriptions = " ".join(
                [description["text"] for description in description_list]
            )

            separated_title_description.append(
                {"title": raw_titles, "description": raw_descriptions}
            )

        result = list()
        for block in separated_title_description:
            title, descriptions = block.values()
            raw_title = title
            entities = education_pipe(raw_title)

            employers = list(filter(lambda x: x["entity_group"] == "ORG", entities))
            employer = employers[0]["word"] if len(employers) > 0 else ""

            dates = list(filter(lambda x: x["entity_group"] == "DATE", entities))
            date = dates[0]["word"] if len(dates) > 0 else ""

            countries = list(filter(lambda x: x["entity_group"] == "LOC", entities))
            country = countries[0]["word"] if len(countries) > 0 else ""

            result.append(
                {
                    "title": raw_title,
                    "employer": employer,
                    "date": date,
                    "country": country,
                    "description": descriptions,
                }
            )

        return result
