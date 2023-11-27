import os
import time
import signal
import requests
from htmldate import find_date

excluded_sites = ["politifact.com",
                  "factcheck.org",
                  "snopes.com",
                  "washingtonpost.com/news/fact-checker/",
                  "apnews.com/hub/ap-fact-check",
                  "fullfact.org/",
                  "reuters.com/fact-check",
                  "youtube.com",
                  ".pdf",
                  "fact-check",
                  "factcheck"
                  ]

PLACE_HOLDER = {
    "entities_info": [
        {
            "name": "placeholder",
            "description": "placeholder"
        }
    ],
    "pages_info": [
        {
            'page_name': 'placeholder',
            'page_url': 'placeholder',
            'page_timestamp': 'placeholder',
            'page_snippet': 'placeholder',
        }
    ]
}


def timeout_handler(num, stack):
    print("received sigalrm")
    raise Exception("Time out!")


class WebRetriever:

    def __init__(self, engine: str, sites_constrain=True, answer_count: int = 10):
        self.engine = engine
        if engine == 'bing':
            self.subscription_key = os.environ[
                'BING_SEARCH_V7_SUBSCRIPTION_KEY']
            self.subscription_key = "4eb55afbffd34ae9943c95a15da944b0"
            self.endpoint = os.environ['BING_SEARCH_V7_ENDPOINT']
            self.mkt = 'en-US'
            self.answer_count = answer_count
            self.headers = {'Ocp-Apim-Subscription-Key': self.subscription_key}
            self.site_constraint = sites_constrain
        elif engine == 'google':
            pass

    def get_results(self, query: str, time_stamp=None, raw_count=20):
        if self.engine == 'bing':
            params = {
                'q': query,
                'mkt': self.mkt,
                'count': raw_count,
                'freshness': "2000-01-01..{}".format(time_stamp) if
                time_stamp else None
            }
            try:
                start = time.time()
                response = requests.get(self.endpoint,
                                        headers=self.headers,
                                        params=params
                                        )
                response.raise_for_status()
                end = time.time()
                print(f'search for query:{query} finished, time take:', end - start)
            except Exception as e:
                print('exception happens during the search:')
                print(e)
                return PLACE_HOLDER
            response_json = response.json()
            entities_info = []
            pages_info = []

            if 'entities' in response_json.keys():
                entities_info = []
                for info in response_json['entities']['value']:
                    entities_info.append(
                        {
                            'name': info['name'],
                            'description': info['description']
                        }
                    )
            if 'webPages' not in response_json:
                return PLACE_HOLDER
            retrieved_pages = response_json['webPages']['value']
            count = 0
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            for i, page in enumerate(retrieved_pages):
                page_info_entry = {
                    'page_name': None,
                    'page_url': None,
                    'page_timestamp': None,
                    'page_snippet': None
                }
                if not self.site_constraint or not self.url_is_excluded(page['url']):
                    page_info_entry['page_name'] = page['name']
                    page_info_entry['page_url'] = page['url']
                    page_info_entry['page_snippet'] = page['snippet']

                    try:
                        # print("getting page timestamp:", page['url'])
                        page_date = find_date(page['url'], verbose=False, original_date=False)
                        page_info_entry['page_timestamp'] = page_date
                        # print(page_date)
                    except Exception as ex:
                        print('Error happens during extracting page timestamp:')
                        print(ex)
                    # assert page_info_entry['page_url'] is not None
                    pages_info.append(page_info_entry)
                    count += 1
                    if count == self.answer_count:
                        break
            return {
                "entities_info": entities_info,
                "pages_info": pages_info
            }

    @staticmethod
    def url_is_excluded(url):
        for ex_url in excluded_sites:
            if ex_url in url:
                return True
        return False

