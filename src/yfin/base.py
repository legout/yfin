from parallel_requests.parallel_requests_niquests import ParallelRequests
from yfinance.data import YfData
import requests

class Session(ParallelRequests):
    def __init__(
        self,
        concurrency: int = 100,
        max_retries: int = 5,
        backoff_factor: int = 0.05,
        backoff_max: int = 10,
        backoff_jitter: bool = 0.1,
        random_proxy: bool = False,
        random_user_agent: bool = True,
        proxies: list | str | None = None,
        user_agents: list | str | None = None,
        cookies: dict | None = None,
        verbose: bool = True,
        debug: bool = False,
        warnings: bool = False,
    ) -> None:
        super().__init__(
            concurrency=concurrency,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            backoff_max=backoff_max,
            backoff_jitter=backoff_jitter,
            random_proxy=random_proxy,
            random_user_agent=random_user_agent,
            proxies=proxies,
            user_agents=user_agents,
            cookies=cookies,
            verbose=verbose,
            debug=debug,
            warnings=warnings,
        )
        self._set_cookie_and_crumb()

    def _get_cookie_and_crumb(self):
        yf = YfData(requests.Session())
        return yf._get_cookie_and_crumb()

    def _set_cookie_and_crumb(self):
        cookies, self.crumb, _ = self._get_cookie_and_crumb()
        self._session.cookies.set_cookie(cookies)
