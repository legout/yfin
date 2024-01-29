from parallel_requests.parallel_requests_asyncer import ParallelRequests
from yfinance.data import YfData


class Session(ParallelRequests):
    def __init__(
        self,
        concurrency: int = 10,
        max_retries: int = 5,
        random_delay_multiplier: int = 1,
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
            random_delay_multiplier=random_delay_multiplier,
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
        yf = YfData(self._session)
        return yf._get_cookie_and_crumb()

    def _set_cookie_and_crumb(self):
        cookies, self.crumb, _ = self._get_cookie_and_crumb()
        self._session.cookies.set_cookie(cookies)
