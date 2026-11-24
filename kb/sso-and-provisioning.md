---
title: Single sign-on and user provisioning
updated: 2026-07-15
---
# Single sign-on and user provisioning

## SAML single sign-on

SAML SSO is available on Business and Enterprise. It works with Okta, Microsoft Entra ID (formerly Azure AD), Google Workspace and any SAML 2.0 identity provider.

### Setting up SAML

1. In Admin console > Security > Single sign-on, copy the ACS URL and Entity ID.
2. Create a SAML app in your identity provider with those values. The NameID must be the user's email address.
3. Upload the identity provider's metadata XML to Kitebox.
4. Test with one account before choosing "Require SSO" for everyone.

### SSO error codes

- SSO-401: the SAML response is signed with a certificate Kitebox doesn't have. Upload the current certificate from your identity provider.
- SSO-403: the email in the SAML response doesn't belong to a verified domain of the workspace. Verify the domain under Admin console > Domains.
- SSO-409: a user with that email already exists in a different Kitebox workspace. They must leave that workspace first.

## SCIM provisioning

SCIM 2.0 provisioning is available on Enterprise only. It creates, updates and deactivates users automatically from Okta or Microsoft Entra ID. Team and Business workspaces add and remove users by hand or by CSV import under Admin console > Users > Import.
