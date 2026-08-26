# OPP-115 -> GDPR Coverage Report

## Category stats (loaded corpus)

| Category | Annotations | Policies | Segments | Generatable | Excluded |
|---|---:|---:|---:|---:|---:|
| Data Retention | 370 | 76 | 156 | 156 | 0 |
| Data Security | 1008 | 102 | 375 | 375 | 0 |
| Do Not Track | 90 | 31 | 32 | 32 | 0 |
| First Party Collection/Use | 8935 | 114 | 1522 | 1522 | 0 |
| International and Specific Audiences | 939 | 90 | 353 | 353 | 0 |
| Other | 3548 | 114 | 1763 | 287 | 1476 |
| Policy Change | 548 | 93 | 192 | 192 | 0 |
| Third Party Sharing/Collection | 5221 | 114 | 1186 | 1186 | 0 |
| User Access, Edit and Deletion | 746 | 90 | 231 | 231 | 0 |
| User Choice/Control | 1789 | 106 | 632 | 632 | 0 |

## Coverage flags

- **Data Retention** [attribute_value_thin]: attribute 'Personal Information Type' value 'Computer information': only 1 occurrence(s) (threshold: 5).
- **Data Retention** [attribute_value_thin]: attribute 'Personal Information Type' value 'Location': only 2 occurrence(s) (threshold: 5).
- **Data Retention** [attribute_value_thin]: attribute 'Personal Information Type' value 'User profile': only 3 occurrence(s) (threshold: 5).
- **Data Retention** [attribute_value_thin]: attribute 'Personal Information Type' value 'Personal identifier': only 2 occurrence(s) (threshold: 5).
- **Data Retention** [attribute_value_thin]: attribute 'Personal Information Type' value 'Social media data': only 2 occurrence(s) (threshold: 5).
- **Data Retention** [attribute_value_thin]: attribute 'Personal Information Type' value 'Demographic': only 1 occurrence(s) (threshold: 5).
- **Data Retention** [attribute_value_thin]: attribute 'Personal Information Type' value 'Health': only 1 occurrence(s) (threshold: 5).
- **Data Retention** [attribute_value_thin]: attribute 'Retention Purpose' value 'Marketing': only 3 occurrence(s) (threshold: 5).
- **Data Retention** [attribute_value_thin]: attribute 'Retention Purpose' value 'Advertising': only 1 occurrence(s) (threshold: 5).
- **Data Security** [attribute_value_thin]: attribute 'Security Measure' value 'Unspecified': only 4 occurrence(s) (threshold: 5).
- **Do Not Track** [not_gdpr_native]: Do Not Track is mapped via a weak analogy, not a genuine GDPR requirement (see mapping config note).
- **Do Not Track** [attribute_value_thin]: attribute 'Do Not Track policy' value 'Mentioned, but unclear if honored': only 3 occurrence(s) (threshold: 5).
- **Do Not Track** [attribute_value_thin]: attribute 'Do Not Track policy' value 'Honored': only 1 occurrence(s) (threshold: 5).
- **First Party Collection/Use** [config_declared_thin]: mapping config declares coverage: thin (legal_basis_granularity: category never records which Art 6(1)(a)-(f) basis applies., Does not distinguish Art 13 (collected from subject) vs Art 14 (collected from other sources) -- both are mapped as primary; build_sft_dataset.py grounds each annotation against both.).
- **First Party Collection/Use** [attribute_value_thin]: attribute 'Action First-Party' value 'Collect on mobile website': only 2 occurrence(s) (threshold: 5).
- **International and Specific Audiences** [config_declared_thin]: mapping config declares coverage: thin (cross_border_transfer_mechanism: Audience Type signals *that* an international audience is addressed, never *how* transfers are safeguarded (adequacy/SCCs/BCRs/derogations)., child_consent_age_verification: 'Children' audience type does not record the stated age threshold or verification mechanism Art 8 requires.).
- **Other** [config_declared_thin]: mapping config declares coverage: thin (dpo_designation: no attribute distinguishes a designated DPO contact (Art 37-39) from a general privacy-inbox address.).
- **Policy Change** [attribute_value_thin]: attribute 'Change Type' value 'Non-privacy relevant change': only 1 occurrence(s) (threshold: 5).
- **Policy Change** [attribute_value_thin]: attribute 'User Choice' value 'Opt-out': only 3 occurrence(s) (threshold: 5).
- **Policy Change** [attribute_value_thin]: attribute 'User Choice' value 'Other': only 4 occurrence(s) (threshold: 5).
- **Third Party Sharing/Collection** [config_declared_thin]: mapping config declares coverage: thin (cross_border_transfer_mechanism: 'Third Party Entity' does not record recipient location, so Chapter V (Art 44-49) transfer-mechanism compliance is never directly assessable from this category alone., Does not distinguish an independent third-party recipient (Art 13(1)(e)) from a processor under Art 28.).
- **Third Party Sharing/Collection** [attribute_value_thin]: attribute 'User Type' value 'User without account': only 3 occurrence(s) (threshold: 5).
- **User Access, Edit and Deletion** [config_declared_thin]: mapping config declares coverage: thin (portability: Art 20 (structured, machine-readable export) is not a distinct OPP-115 Access Type -- 'View' covers both plain access and portable export., restriction_of_processing: Art 18 (right to restrict rather than erase) has no corresponding Access Type at all.).
- **User Access, Edit and Deletion** [attribute_value_thin]: attribute 'Access Type' value 'Export': only 1 occurrence(s) (threshold: 5).
- **User Access, Edit and Deletion** [attribute_value_thin]: attribute 'Access Type' value 'None': only 4 occurrence(s) (threshold: 5).
- **User Choice/Control** [config_declared_thin]: mapping config declares coverage: thin (Cannot distinguish 'withdraw consent' (Art 7(3)) from 'object to legitimate-interest processing' (Art 21) -- both surface as generic opt-out attributes in OPP-115.).
- **User Choice/Control** [attribute_value_thin]: attribute 'Purpose' value 'Merger/Acquisition': only 3 occurrence(s) (threshold: 5).
- **User Choice/Control** [attribute_value_thin]: attribute 'Personal Information Type' value 'Computer information': only 3 occurrence(s) (threshold: 5).
- **User Choice/Control** [attribute_value_thin]: attribute 'Personal Information Type' value 'Survey data': only 2 occurrence(s) (threshold: 5).
- **User Choice/Control** [attribute_value_thin]: attribute 'Personal Information Type' value 'Financial': only 4 occurrence(s) (threshold: 5).
- **User Choice/Control** [attribute_value_thin]: attribute 'Personal Information Type' value 'Personal identifier': only 2 occurrence(s) (threshold: 5).
- **User Choice/Control** [attribute_value_thin]: attribute 'Personal Information Type' value 'Demographic': only 4 occurrence(s) (threshold: 5).
- **User Choice/Control** [attribute_value_thin]: attribute 'Personal Information Type' value 'Social media data': only 3 occurrence(s) (threshold: 5).
- **User Choice/Control** [attribute_value_thin]: attribute 'Personal Information Type' value 'IP address and device IDs': only 1 occurrence(s) (threshold: 5).

## GDPR schema gaps (need new labeled data, not more OPP-115)

### legal_basis_granularity (Art. 6(1)(a), 6(1)(b), 6(1)(c), 6(1)(d), 6(1)(e), 6(1)(f))

Which specific Art 6(1)(a)-(f) legal basis applies to a given processing purpose.

*Why OPP-115 can't capture this:* No OPP-115 category or attribute records legal basis at all; "Purpose" (First Party Collection/Use) and "Choice Type" attributes are the nearest proxies but conflate basis with disclosed purpose/choice mechanism, not the basis itself.

### special_category_data (Art. 9)

Additional Art 9 condition required for special-category data (health, biometric, religious, etc.).

*Why OPP-115 can't capture this:* "Personal Information Type" has no special-category-specific value set distinguishing Art 9 data from ordinary personal data.

### dpo_designation (Art. 37, 38, 39)

Whether a Data Protection Officer is designated, and their contact details/tasks.

*Why OPP-115 can't capture this:* No attribute distinguishes a DPO contact from a general privacy-inbox contact (see Other category above).

### cross_border_transfer_mechanism (Art. 44, 45, 46, 47, 49)

The specific transfer safeguard used for international transfers: adequacy decision, SCCs, BCRs, or a derogation.

*Why OPP-115 can't capture this:* 'Audience Type' records that an international audience/transfer is addressed, never the mechanism.

### dpia_requirement (Art. 35)

Whether a Data Protection Impact Assessment was conducted for high-risk processing.

*Why OPP-115 can't capture this:* No OPP-115 category corresponds to DPIA/high-risk-processing assessment at all.

### privacy_by_design_default (Art. 25)

Data protection by design and by default in system/product design.

*Why OPP-115 can't capture this:* No OPP-115 category records design-level/default-setting practices, only disclosed data-handling practices.

### automated_decision_making (Art. 22)

Right not to be subject to a decision based solely on automated processing, including profiling.

*Why OPP-115 can't capture this:* No OPP-115 category or attribute addresses automated decision-making/profiling logic.

### restriction_of_processing (Art. 18)

Right to restrict (rather than erase) processing.

*Why OPP-115 can't capture this:* 'Access Type' in User Access, Edit and Deletion has no restriction-of-processing value distinct from erasure.

### breach_notification_specifics (Art. 33, 34)

72-hour supervisory-authority notification timeline and high-risk data-subject notification duty.

*Why OPP-115 can't capture this:* OPP-115 has no breach-notification category; Data Security's 'Security Measure' attribute covers preventive measures, not breach response.

### portability (Art. 20)

Right to receive personal data in a structured, commonly used, machine-readable format and transmit it to another controller.

*Why OPP-115 can't capture this:* 'Access Type: View' conflates plain access/export with a portable, machine-readable export.
