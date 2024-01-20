import os
import time
import pandas as pd
import openai

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


class Utility:
    def import_api_key():
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        if not OPENAI_API_KEY:
            raise ValueError("No OpenAI API key found!")

        print("OPENAI API KEY = ", OPENAI_API_KEY)

    @staticmethod
    def extract_type_from_response(response):
        for c in response[::-1]:
            if c == "@" or c == "#":
                print("Inside last search!")
                return c

        raise Exception(f"No @ or # found in response = {response}")

    @staticmethod
    def calculate_metrics(ground_truth, predicted):
        clsf_report = classification_report(
            y_true=ground_truth, y_pred=predicted, output_dict=True
        )
        cf_matrix = confusion_matrix(ground_truth, predicted)

        precision = clsf_report["weighted avg"]["precision"]
        recall = clsf_report["weighted avg"]["recall"]
        f1 = clsf_report["weighted avg"]["f1-score"]
        accuracy = accuracy_score(ground_truth, predicted)

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Confusion Matrix": cf_matrix,
        }

    @staticmethod
    def get_tweet_data(file_name):
        df = pd.read_csv(file_name, index_col=0)
        return df

    @staticmethod
    def write_prediction_output(tweet_objects, file_name_to_write):
        tweet_objects.to_csv(file_name_to_write)

    @staticmethod
    def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message["content"]


class Default(dict):
    def __missing__(self, key):
        return f"{{{key}}}"


class ClaimExistence:
    INPUT_VARIABLES = ["delimiter", "tweet"]

    DELIMITER = "```"

    def __init__(self, model_name, input_file_name):
        (
            self.system_message,
            self.tweet_objects,
        ) = ClaimExistence.generate_system_prompt_for_claim_existence(input_file_name)

        self.model_name = model_name

    def does_tweet_contain_claim(self, tweet):
        messages = [
            {"role": "system", "content": self.system_message},
            {
                "role": "user",
                "content": f"Tweet = {Category.DELIMITER}{tweet}{Category.DELIMITER}",
            },
        ]

        response = Utility.get_completion_from_messages(
            messages, self.model_name, temperature=0.2
        )

        print("Response = ", response)

        response_number = Utility.extract_type_from_response(response)

        return response_number

    def generate_claim_existence_metrics(
        self, output_file_name, tweet_content_column="polished_text"
    ):
        claim_existence_ground_truths = []
        claim_existence_predicted_outputs = []

        print(
            "<======= Generating metrics for Claim Existence =======>",
        )
        print()

        claim_existence_prediction_column_name = f"{self.model_name}-predicted-claim"

        if claim_existence_prediction_column_name not in self.tweet_objects:
            self.tweet_objects[claim_existence_prediction_column_name] = -1

        for index, row in self.tweet_objects.iterrows():
            print("Processing tweet with index# =", index)
            tweet = row[tweet_content_column]
            print("Tweet content = ", tweet)

            for _ in range(10):
                try:
                    claim_existence_ground_truth = int(row["Claim"])
                except:
                    continue
                else:
                    break

            if row[claim_existence_prediction_column_name] != -1:
                claim_existence_ground_truths.append(claim_existence_ground_truth)
                claim_existence_predicted_outputs.append(
                    int(row[claim_existence_prediction_column_name])
                )
                continue

            for _ in range(10):
                try:
                    claim_existence_predicted_output = self.does_tweet_contain_claim(
                        tweet
                    )

                    claim_existence_predicted_output = (
                        0 if claim_existence_predicted_output == "@" else 1
                    )

                    if claim_existence_predicted_output is None:
                        continue
                    else:
                        break
                except:
                    continue

            if claim_existence_predicted_output is None:
                print("None for index# = ", index, "and tweet content =", tweet)
                raise Exception("Did not get predicted output for tweet")

            if index > 0 and index % 5 == 0:
                print(
                    "Metrics till now =",
                    Utility.calculate_metrics(
                        claim_existence_ground_truths, claim_existence_predicted_outputs
                    ),
                )

            claim_existence_ground_truths.append(claim_existence_ground_truth)
            claim_existence_predicted_outputs.append(claim_existence_predicted_output)

            self.tweet_objects.loc[
                index, claim_existence_prediction_column_name
            ] = claim_existence_predicted_output

            print(
                "Ground truth =",
                claim_existence_ground_truth,
                "Predicted output =",
                claim_existence_predicted_output,
            )
            print("Finished Processing tweet with index# =", index)
            print()

            Utility.write_prediction_output(self.tweet_objects, output_file_name)

            time.sleep(0.1)

        print("<======= Finished generating metrics for claim existence =======>")

        print("Ground truths = ", claim_existence_ground_truths)
        print("Predictions = ", claim_existence_predicted_outputs)

        Utility.write_prediction_output(self.tweet_objects, output_file_name)

        return Utility.calculate_metrics(
            claim_existence_ground_truths, claim_existence_predicted_outputs
        )

    @staticmethod
    def generate_system_prompt_for_claim_existence(input_file_name):
        tweet_examples_for_claim_existence = """
        Some examples of tweets that contain scientific claims about COVID-19 (expected response #):
            a) "It changed on 19 July, Tristan and I was just pointing out that your tweet was incorrect. The facts are: vaccinated visitors from UK to France do not need a negative Covid test nor quarantine. unvaccinated visitors from UK to France need only a negative Covid test."
            b) "Arunachal Pradeshs Covid19 tally rises to 15,484"
            c) "The Covid19 front page. Shows how serious the pandemic is. Corona is real. Take care; observe social distancing, wear face mask, washing hands with running water."
        """

        tweet_examples_for_non_claim_existence = """
        Some examples of tweets that do not contain scientific claims about COVID-19 (expected response @):
            a) "Are footballers in west Hull flouting lockdown rules?"
            b) "Iranian pharmaceutical company Shifa Pharmed has begun registering volunteers for human trials of the countrys first domestic Covid19 vaccine candidate"
            c) "The Centre is gearing up for the roll out of COVID19 vaccine across the country, with four States all set to initiate a dryrun for vaccine administration next week, the Union Health Ministry said."
        """

        system_message = """
        Imagine you're a COVID-19 tweets classifier. You need to determine whether tweets contain a scientific claim about COVID-19.
        
        The tweets will be delimited with {delimiter} characters.
        
        Use the following guidelines to make your decision:
            1. Direct statements about the COVID-19 virus, including its origin, transmission methods, prevention methods, or symptoms are considered as claims.
            2. Opinionated, anecdotal, or hearsays about COVID-19 topics may contain claim.
            3. Reports on COVID-19 cases, COVID-19-related deaths, or instances of someone testing positive for COVID-19 are claims.
            4. Someone making an observation is not a claim.
            5. The impact of COVID-19 on fields other than science, such as business, law, history, politics, and operations, is not considered a claim.
            9. The claim may be scientifically verifiable, references to COVID-19 scentific studies or mentions COVID-19 scientific topics in general.
            
        {tweet_examples_for_claim_existence}
        
        {tweet_examples_for_non_claim_existence}

        If the tweet is scientifically verifiable, return #. Otherwise, return @.
        
        This task is very important to my career. You'd better be sure. Make sure to take another look at your response before responding.
        """.format_map(
            Default(
                tweet_examples_for_claim_existence=tweet_examples_for_claim_existence,
                tweet_examples_for_non_claim_existence=tweet_examples_for_non_claim_existence,
            )
        )

        tweet_objects = Utility.get_tweet_data(input_file_name)

        print("Length of tweet objects = ", len(tweet_objects))

        return system_message, tweet_objects


class Category:
    INPUT_VARIABLES = ["delimiter", "tweet"]

    DELIMITER = "```"

    CATEGORY_DESCRIPTIONS = {1: "Scientifically Verifiable"}

    def __init__(self, category_type, model_name):
        self.category_type = category_type
        self.model_name = model_name

    def does_tweet_fall_into_category(self, tweet):
        messages = [
            {"role": "system", "content": self.system_message},
            {
                "role": "user",
                "content": f"Tweet = {Category.DELIMITER}{tweet}{Category.DELIMITER}",
            },
        ]

        response = Utility.get_completion_from_messages(
            messages, self.model_name, temperature=0
        )

        print("Response = ", response)

        response_number = Utility.extract_type_from_response(response)

        return response_number

    def generate_cat_metrics(
        self, output_file_name, tweet_content_column="polished_text"
    ):
        category_ground_truths = []
        category_predicted_outputs = []

        print(
            "<======= Generating metrics for category type =",
            self.category_type,
            "=======>",
        )
        print()

        category_type_prediction_column_name = (
            f"{self.model_name}-predicted-cat{self.category_type}"
        )

        if category_type_prediction_column_name not in self.tweet_objects:
            self.tweet_objects[category_type_prediction_column_name] = -1

        for index, row in self.tweet_objects.iterrows():
            print("Processing tweet with index# =", index)
            tweet = row[tweet_content_column]
            print("Tweet content = ", tweet)

            for _ in range(10):
                try:
                    category_ground_truth = int(row[f"cat{self.category_type}"])
                except:
                    continue
                else:
                    break

            if row[category_type_prediction_column_name] != -1:
                category_ground_truths.append(category_ground_truth)
                category_predicted_outputs.append(
                    int(row[category_type_prediction_column_name])
                )
                continue

            claim_existence_prediction_column_name = (
                f"{self.model_name}-predicted-claim"
            )

            claim_existence_predicted_output = int(
                row[claim_existence_prediction_column_name]
            )

            if claim_existence_predicted_output == 0:
                category_predicted_output = 0
            else:
                for _ in range(10):
                    try:
                        category_predicted_output = self.does_tweet_fall_into_category(
                            tweet
                        )

                        category_predicted_output = (
                            0 if category_predicted_output == "@" else 1
                        )

                        if category_predicted_output is None:
                            continue
                        else:
                            break
                    except:
                        continue

            if category_predicted_output is None:
                print("None for index# = ", index, "and tweet content =", tweet)
                raise Exception("Did not get predicted output for tweet")

            if index > 0 and index % 5 == 0:
                print(
                    "Metrics till now =",
                    Utility.calculate_metrics(
                        category_ground_truths, category_predicted_outputs
                    ),
                )

            category_ground_truths.append(category_ground_truth)
            category_predicted_outputs.append(category_predicted_output)

            self.tweet_objects.loc[
                index, category_type_prediction_column_name
            ] = category_predicted_output

            print(
                "Ground truth =",
                category_ground_truth,
                "Predicted output =",
                category_predicted_output,
            )
            print("Finished Processing tweet with index# =", index)
            print()

            Utility.write_prediction_output(self.tweet_objects, output_file_name)

            time.sleep(0.1)

        print("<======= Finished generating metrics for claim existence =======>")

        print("Ground truths = ", category_ground_truths)
        print("Predictions = ", category_predicted_outputs)

        Utility.write_prediction_output(self.tweet_objects, output_file_name)

        return Utility.calculate_metrics(
            category_ground_truths, category_predicted_outputs
        )


class Category1(Category):
    CATEGORY_TYPE = 1
    CATEGORY_DESCRIPTION = ""

    def __init__(self, model_name, input_file_name):
        (
            self.system_message,
            self.tweet_objects,
        ) = Category1.generate_system_prompt_for_category1(input_file_name)

        super().__init__(Category1.CATEGORY_TYPE, model_name)

    @staticmethod
    def generate_system_prompt_for_category1(input_file_name):
        tweet_examples_of_category1 = """
        Some examples of tweets that ARE scientifically verifiable (expected response #):
            a) " ::people_holding_hands:: We can now meet our family and friends outdoors in a group of 6, or 2 households ::leftright_arrow:: Its important that when we do, we follow social distancing guidance ::backhand_index_pointing_right:: This will help to stop the spread of COVID19 as we take the next step out of lockdown LetsDoItForLancashire "
            b) ": BREAKING: Dozens of cops in Massachusetts have resigned in protest of the vaccine mandates. TO WISH THEM GOOD RIDDA"
            c) ": BREAKING Syria president and first lady test positive for COVID19: presidency AFP"
        """

        tweet_examples_of_non_category1 = """
        Some examples of tweets ARE NOT scientifically verifiable (expected response @):
            a) " : The ones calling for lockdown, without risk or injury to themselves, should pay up."
            b) ": Can you catch coronavirus from handling cash? A new study says the risk is low"
            c) ": I wouldnt trust anything this man touches. NoVaccineForMe"
        """

        system_message = """
        Imagine you're a COVID-19 tweets classifier. You need to determine whether tweets fall into scientifically verifiable claim category.
        
        The tweets will be delimited with {delimiter} characters.

        A claim or a question is scientifically verified if it's scientifically shown to be true or scientifically shown to be false.
        
        Use the following guidelines to make your decision:
            1. Direct statements about the COVID-19 virus, including its origin, transmission methods, prevention methods, or symptoms, are scientifically verifiable.
            2. Opinionated, anecdotal, or hearsay claims about COVID-19 topics may be scientifically verifiable.
            3. Reports on COVID-19 cases, COVID-19-related deaths, or instances of someone testing positive for COVID-19 are scientifically verifiable.
            4. Someone making an observation is not scientifically verifiable.
            5. The impact of COVID-19 on fields other than science, such as business, law, history, politics, and operations, is not scientifically verifiable.
            6. Second-hand opinions or queries about COVID-19 topics are not scientifically verifiable.
            7. Asking questions about COVID-19 topics without making a direct claim is not scientifically verifiable.
            8. Instructions, information, notifications, or announcements about COVID-19 topics that do not include opinions are not scientifically verifiable.
            9. Political, business, or legal motives behind COVID-19 topics are not scientifically verifiable.
            10. Tweets containing phrases like "Read the whole story here," "Full version," "This story from," "Live video," "How it became," "Here's a quick look," etc., are not scientifically verifiable.
            
        {tweet_examples_of_category1}
        
        {tweet_examples_of_non_category1}

        If the tweet is scientifically verifiable, return #. Otherwise, return @.
        
        This task is very important to my career. You'd better be sure. Make sure to take another look at your response before responding.
        """.format_map(
            Default(
                tweet_examples_of_non_category1=tweet_examples_of_non_category1,
                tweet_examples_of_category1=tweet_examples_of_category1,
            )
        )

        tweet_objects = Utility.get_tweet_data(input_file_name)

        print("Length of tweet objects = ", len(tweet_objects))

        return system_message, tweet_objects


# def does_tweet_contain_claim(tweet, delimiter="```"):
#     system_message = f"""
#     You will be provided with some tweets. The tweets will be delimited with {delimiter} characters.

#     Your task is to determine whether the tweets contain a science related claim.

#     1. The tweet needs to be related to science. If the tweet is not related to science, you don't need to evaluate further.
#         - Example: "I heard that your lung capacity could be severly damaged if you catch COVID-19."
#         - Counterexample: "COVID-19 pandemic has severely disrupted the global supply chain."

#     2. The tweet contains a speculation or hypothesis (could be an inference with or without some numeric data) about a scientific topic that can be verified in any of the following way:
#         I. It has already been verified scientifically and could be found in scientific documents or research papers.
#             - Example: "COVID-19 primarily spreads via respiratory droplets when an infected person coughs, sneezes, talks, or breathes."
#         II. It could be scientifically verified in theory.
#         III. The claim can be unsubstantiated. There may not have been adequate research to accept or deny the claim.
#             - Example: "The people who have caught COVID-19 will live a shorter life than those who haven't." (Note: We don't yet know the effects of COVID-19 on lifespan, but in the future there could be studies that reveal these effects.)
#         IV. The claims can be absurd and there could be no way of verifying them.
#             - Example: "I think the aliens spread COVID-19."

#     3. The tweet will not be considered to contain a scientific claim if:
#         I. It does not contain a claim related to science but related to some other domain such as Politics, Business or Technology.
#         II. No claim could be inferred from the tweet or the tweet contains only facts.
#         III. It contains claims shaping from individuals or organizations biases which are unrelated to science.
#             - Example: "I think it's not empathetic to disregard mental health issues emerging from COVID-19".

#     Provide your response as an integer. If the tweet contains a science related claim, your response should be 1. Otherwise, it should be 0.

#     Your response should only include an integer and nothing else.
#     """
#     messages = [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": f"{delimiter}{tweet}{delimiter}"},
#     ]

#     return get_completion_from_messages(messages)


def main():
    Utility.import_api_key()

    model_name = "gpt-4-1106-preview"

    input_file_name = "gpt-tweets.csv"
    output_file_name = "gpt-tweets.csv"

    # claim_existence = ClaimExistence(model_name, input_file_name)
    # claim_existence.generate_claim_existence_metrics(output_file_name)

    cat1 = Category1(model_name, input_file_name)
    cat1.generate_cat_metrics(output_file_name)


if __name__ == "__main__":
    main()
