# User Management System

## API Endpoints

### Authentication (Public)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login and get tokens |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| POST | `/api/v1/auth/logout` | Revoke refresh token |

### User Profile (Authenticated)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/users/me` | Get profile |
| PATCH | `/api/v1/users/me` | Update profile |
| POST | `/api/v1/users/me/password` | Change password |
| DELETE | `/api/v1/users/me` | Soft delete account |
| GET | `/api/v1/users/me/stats` | Get usage statistics |
| GET | `/api/v1/users/me/jobs` | List generation jobs |
| POST | `/api/v1/users/me/logout-all` | Logout all devices |

## Usage Examples

### Register

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123", "display_name": "John"}'
```

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123"}'
```

### Authenticated Request

```bash
curl http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer <access_token>"
```

### Refresh Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<refresh_token>"}'
```

## Security Notes

1. **Token Rotation**: Each refresh creates a new refresh token. Old tokens are invalidated.

2. **Family Tracking**: If a revoked refresh token is reused, ALL tokens in that family are revoked (potential theft).

3. **Password Change**: Changes password and revokes all refresh tokens.

4. **Soft Delete**: Accounts are deactivated, not deleted. Implement cleanup task later.

## Adding Auth to Existing Routes

To protect existing routes, add the guard:

```python
from src.api.security import auth_guard

class MyController(Controller):
    guards = [auth_guard]
    
    @get("/protected")
    async def protected_route(self, request: Request) -> ...:
        user_id = request.state["user_id"]  # UUID from token
        ...
```

## Database Schema

```sql
-- users table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    subscription_tier VARCHAR(20) DEFAULT 'free',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- refresh_tokens table
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    family_id UUID NOT NULL,
    user_agent TEXT,
    ip_address VARCHAR(45),
    is_revoked BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);
```

## Future Enhancements

- [X] Add FK constraints from existing tables to users (see migration comments)
- [ ] Implement background task for expired token cleanup
- [ ] Add rate limiting for auth endpoints
- [ ] Implement PKCE flow for OAuth2 clients
- [ ] Add email verification
- [ ] Add account recovery flow
