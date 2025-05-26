from fastapi import HTTPException, status
from typing import Any, Dict, Optional

class APIError(HTTPException):
    """
    API 错误基类
    """
    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class NotFoundError(APIError):
    """
    资源未找到错误
    """
    def __init__(
        self,
        detail: Any = "请求的资源不存在",
        headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            headers=headers,
        )


class ValidationError(APIError):
    """
    数据验证错误
    """
    def __init__(
        self,
        detail: Any = "数据验证失败",
        headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            headers=headers,
        )


class AuthenticationError(APIError):
    """
    认证错误
    """
    def __init__(
        self,
        detail: Any = "认证失败",
        headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers=headers,
        )


class AuthorizationError(APIError):
    """
    授权错误
    """
    def __init__(
        self,
        detail: Any = "没有权限执行此操作",
        headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            headers=headers,
        )


class ServerError(APIError):
    """
    服务器错误
    """
    def __init__(
        self,
        detail: Any = "服务器内部错误",
        headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            headers=headers,
        )


class ModelError(APIError):
    """
    模型错误
    """
    def __init__(
        self,
        detail: Any = "模型处理失败",
        headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            headers=headers,
        )
